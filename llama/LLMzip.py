from typing import List
import pickle, io, os
import torch
from llama.tokenizer import Tokenizer
from llama.model import Transformer
from llama.llmzip_utils import *
from AC.arithmeticcoding import *
import numpy as np
import pandas as pd
import zlib
import sys
import binascii
import json
from llama.llmzip_utils import build_compressed_pkl_name

class LLMzip_encode:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, use_hf: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.AC_encoder = None
        self.compression_alg = None
        self.use_hf = use_hf
        
    def encode_batch(self, prompt_tokens):
        bsz = prompt_tokens.shape[0]
        prompt_size = prompt_tokens.shape[1]
        
        # bsz*prompt_size大小的张量全部初始化为填充token id再赋值
        if self.use_hf:
            tokens = torch.full((bsz, prompt_size), -1).cuda().long()
            # tokens = torch.full((bsz, prompt_size), self.tokenizer.pad_token_id).cuda().long()
        else:
            params = self.model.params
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
            assert prompt_size <= params.max_seq_len, (prompt_size,params.max_seq_len)
            tokens = torch.full((bsz, prompt_size), self.tokenizer.pad_id).cuda().long()
        tokens[:bsz, : prompt_size] = torch.tensor(prompt_tokens).long()


        cur_pos = prompt_size-1
        prev_pos = 0
        
        
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits, dim=-1)
        rank = gen_rank(probs,next_token=tokens[:,cur_pos])
        
        probs_np2 = probs.cpu().numpy()
        tokens_np2 = tokens[:,cur_pos].cpu().numpy() # 当前token id
        ranks_np2 = rank.cpu().numpy() # 当前tokenid在prob概率列表中的排名
        # if self.use_hf:
        #     probs_np2 = probs_np2[0]
        probs_tok = probs_np2[np.arange(bsz),tokens_np2] # 当前token id对应的预测概率
        
        #  cumul 数组的形式为 [0, c1, c1+c2, ..., sum(all)]，表示每个 token 对应的概率区间
        if (self.compression_alg == 'ArithmeticCoding')or(self.compression_alg =='both'):
            cumul = np.zeros(self.model.vocab_size+1, dtype = np.uint64) # 初始化累积分布数组（长度为词表大小+1，用于算术编码）
            for j in range(bsz):
                prob1 = probs_np2[j] # 降低一个维度
                cumul[1:] = np.cumsum(prob1*10000000 + 1) # 概率扩大1000w倍以转化为整数，加一防止概率为0，计算累积和，得到累积分布
                self.AC_encoder.write(cumul, tokens_np2[j])
        return ranks_np2,probs_tok

    def encode_from_tokens(
        self,
        win_size: int,
        compression_alg: str ='ArithmeticCoding',
        compressed_file_name: str = 'LLMzip',
        tokens_full = None,
        batched_encode = False,
        with_context_start=False,
        out_dir: str = None):
        self.win_size = win_size
        self.compression_alg = compression_alg
        self.compressed_file_name = compressed_file_name
        # self.with_context_start
        # 额外+1用于传递目标token
        win_size_enc = win_size + 1 # additional 1 is to pass the true token apart from the context of win_size
        
        n_runs = tokens_full.size-win_size_enc+1
        
        if not with_context_start:
            tokens_encoded = tokens_full
            starter_tokens = None
        else:
            tokens_encoded = tokens_full[win_size:win_size+n_runs]
            starter_tokens = tokens_full[:win_size]
        
        self.N_T = tokens_encoded.size

        # 初始化算术编码器
        if (self.compression_alg == 'ArithmeticCoding')or(self.compression_alg =='both'):
            # self.AC_file_name = compressed_file_name+'_AC.txt'
            self.AC_file_name = build_compressed_pkl_name(self.compressed_file_name, out_dir, win_len=getattr(self, 'win_size', 0), N_T=getattr(self, 'N_T', 0), prefix=os.path.splitext(os.path.basename(self.compressed_file_name))[0])
            file_out = open(self.AC_file_name, 'wb')
            bitout = BitOutputStream(file_out)
            self.AC_encoder = ArithmeticEncoder(32, bitout)
        
        ranks_list = []
        probs_tok_list = []

        if not with_context_start:
            # Running LLM for the starter tokens
            for t_ind in range(1,win_size_enc):
                if self.use_hf:
                    tokens_in = np.array([[self.tokenizer.bos_token_id]+tokens_full[:t_ind].tolist()])
                else:
                    tokens_in = np.array([[self.tokenizer.bos_id]+tokens_full[:t_ind].tolist()])
                ranks,probs_tok = self.encode_batch(tokens_in)
                ranks_list += [ranks]
                probs_tok_list += [probs_tok]
            starter_tokens = None
        else:
            tokens_encoded = tokens_full[win_size:win_size+n_runs]
            starter_tokens = tokens_full[:win_size]
        if batched_encode:
            bsz = self.model.params.max_batch_size
        else:
            bsz = 1

        n_batches = np.ceil(n_runs/bsz).astype(int)

        for b_ind in range(n_batches):

            batch_range_start = b_ind*bsz
            batch_range_stop = np.minimum(n_runs,(b_ind+1)*bsz)
            tokens_batch = np.array([tokens_full[i:i+win_size_enc]for i in range(batch_range_start,batch_range_stop)])
            ranks,probs_tok = self.encode_batch(tokens_batch)
            ranks_list += [ranks]
            probs_tok_list += [probs_tok]
            
            
            # if (b_ind*bsz*100/n_batches)%10 == 0:
            #     print(f'Encoder: Completed {int(b_ind*bsz*100/n_batches)} %')
            
        ranks_full = np.concatenate(ranks_list,0).squeeze()
        probs_tok_full = np.concatenate(probs_tok_list,0).squeeze()
        
        if (self.compression_alg == 'ArithmeticCoding')or(self.compression_alg =='both'):
            self.AC_encoder.finish()
            bitout.close() # 写入AC.txt文件里面
            file_out.close()
            
        if (self.compression_alg == 'RankZip')or(self.compression_alg =='both'):
            str_ranks = get_str_array(ranks_full)
            rank_bytes = bytes(str_ranks,'ascii')
            ranks_comp = zlib.compress(rank_bytes,9)
            
            self.RZ_file_name = compressed_file_name+'_RZ.txt'

            with open(self.RZ_file_name,'wb') as file_out_zip:
                file_out_zip.write(ranks_comp)
        print(f"[LLMzip] Saved compressed binary (pickled) to: {self.AC_file_name}")
        N_C, N_T, Bits = self.compute_compression_ration_pkl(tokens_encoded,probs_tok_full,starter_tokens)
        return N_C, N_T, Bits, self.AC_file_name
    
    def compute_compression_ratio(self,tokens_encoded,probs_tok,starter_tokens):
        text_encoded = self.tokenizer.decode(tokens_encoded.squeeze().tolist())
        
        # if self.starter_tokens != None:
        #     starter_text_encoded = self.tokenizer.decode(starter_tokens)
        #     with open(self.compressed_file_name+'_starter_text.txt','w') as text_file:
        #         text_file.write(text_encoded)
        
        N_T = tokens_encoded.size # 原文经过编码器得到的embedding长度
        N_C = len(text_encoded) # 原文本体的len
        
        df_out = {}
        df_out['$N_C$'] = [N_C]
        df_out['$N_T$'] = [N_T]
        df_out['$H_{ub}$'] = [str(np.sum(-1*np.log2(probs_tok))/N_C)]

        if (self.compression_alg == 'RankZip')or(self.compression_alg =='both'):
            with open(self.RZ_file_name, 'rb') as file_RZ:
                ranks_compressed_bytes = file_RZ.read()
            rho_RZ = len(ranks_compressed_bytes)*8/N_C
            print(f'Compression Ratio for RankZip :  {rho_RZ} bits/char')
            
            df_out['Llama+zlib compressed file size'] = [len(ranks_compressed_bytes)*8]
            df_out['$\rho_{LLaMa+Zlib}$'] = [rho_RZ]
            
        df_out['$\rho_{TbyT}$'] = [str(np.sum(np.ceil(-1*np.log2(probs_tok)))/N_C)]
        
        if (self.compression_alg == 'ArithmeticCoding')or(self.compression_alg =='both'):
            b_ind = 1
            file_in = open(self.AC_file_name, 'rb')
            bitin = BitInputStream(file_in)
            compressed_bits = read_bitstream(bitin)
            rho_AC = compressed_bits.size/N_C
            print(f'Compression Ratio for Arithmetic Coding :  {rho_AC} bits/char')
            file_in.close()
            
            df_out['Llama+AC compressed file size'] = [compressed_bits.size]
            df_out['$\rho_{LLaMa+AC}$'] = [rho_AC]
            
        print(df_out)
            
        with open(self.compressed_file_name+'_metrics.json', 'w') as file_metrics: 
            json.dump(df_out, file_metrics)

    def compute_compression_ration_pkl(self, tokens_encoded, probs_tok, starter_tokens):
        text_encoded = self.tokenizer.decode(tokens_encoded.squeeze().tolist())
        N_T = tokens_encoded.size  # 原文经过编码器得到的embedding长度
        N_C = len(text_encoded)    # 原文本体的长度（字符数）
        df_out = {}
        df_out['$N_C$'] = [N_C]
        df_out['$N_T$'] = [N_T]
        df_out['$H_{ub}$'] = [str(np.sum(-1*np.log2(probs_tok))/N_C)]
        if (self.compression_alg == 'RankZip') or (self.compression_alg == 'both'):
            with open(self.RZ_file_name, 'rb') as file_RZ:
                ranks_compressed_bytes = file_RZ.read()
            rho_RZ = len(ranks_compressed_bytes) * 8 / N_C
            print(f'Compression Ratio for RankZip :  {rho_RZ} bits/char')
            df_out['Llama+zlib compressed file size'] = [len(ranks_compressed_bytes)*8]
            df_out['$\rho_{LLaMa+Zlib}$'] = [rho_RZ]
        df_out['$\rho_{TbyT}$'] = [str(np.sum(np.ceil(-1*np.log2(probs_tok))) / N_C)]

        if (self.compression_alg == 'ArithmeticCoding') or (self.compression_alg == 'both'):
            # 读取由 BitOutputStream 产生的原始二进制文件内容
            # self.AC_file_name 在 encode_from_tokens 时已被创建并写入
            with open(self.AC_file_name, 'rb') as file_in:
                file_bytes = file_in.read()
            # 使用现有的函数来计算 bit-array
            bitin = BitInputStream(io.BytesIO(file_bytes))
            compressed_bits = read_bitstream(bitin)
            rho_AC = compressed_bits.size / N_C
            print(f'Compression Ratio for Arithmetic Coding :  {rho_AC} bits/char')

            df_out['Llama+AC compressed file size'] = [compressed_bits.size]
            df_out['$\rho_{LLaMa+AC}$'] = [rho_AC]

            # # 压缩的字节流保存为 .pkl，并把 win/N_T 放入文件名
            # try:
            #     # build_compressed_pkl_name(input_filepath, out_dir, win_len, N_T, prefix=None)
            #     # 我们把 prefix 设为 self.compressed_file_name 保持与原来名称关联
                
            #     # out_dir 设置为 None -> 会使用当前路径的 basename
            #     pkl_name = build_compressed_pkl_name(self.compressed_file_name, out_dir='.', win_len=getattr(self, 'win_size', 0), N_T=getattr(self, 'N_T', N_T), prefix=os.path.splitext(os.path.basename(self.compressed_file_name))[0])
            # except Exception:
            #     # 兼容兜底：构造文件名
            #     pkl_name = f"{self.compressed_file_name}_win{getattr(self,'win_size',0)}_NT{getattr(self,'N_T',N_T)}.pkl"

            # # 保存原始 bytes（不是 numpy bits），以便 decode 时直接回放原始 bit-stream
            # with open(pkl_name, 'wb') as pf:
            #     pickle.dump(file_bytes, pf)

        print("Compression summary (df_out):")
        print(df_out)
        return N_C, N_T, compressed_bits.size


class LLMzip_decode:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, use_hf: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.raw_decoded_tokens = []  # 新增：存储原始算术解码结果
        self.all_probability_distributions = []  # 新增：存储所有概率分布
        self.use_hf = use_hf
    def decode_AC(
        self,
        win_size,
        starter_tokens,
        total_length: int,
        compressed_file_name: str = 'LLMzip_AC.txt',
        save_raw_results: bool = True,  # 新增参数：是否保存原始结果
        raw_output_file: str = None):   # 新增参数：原始结果输出文件
        
        # 初始化存储
        if save_raw_results:
            self.raw_decoded_tokens = []
            self.all_probability_distributions = []

        # 检查是否为 .pkl（若是则加载 bytes 并用 BytesIO 包装）
        if isinstance(compressed_file_name, str) and compressed_file_name.endswith('.pkl'):
            # load pickled bytes
            with open(compressed_file_name, 'rb') as pf:
                file_bytes = pickle.load(pf)
            bitin = BitInputStream(io.BytesIO(file_bytes))
        else:
            # 兼容老流程：直接以二进制流打开（比如 'LLMzip_AC.txt'）
            file_in = open(compressed_file_name, 'rb')
            bitin = BitInputStream(file_in)

        # file_in = open(compressed_file_name, 'rb')
        # bitin = BitInputStream(file_in)
        dec = ArithmeticDecoder(32, bitin)
        
        bsz = 1    # predicts 1 token at a time

        if starter_tokens is not None:
            total_length += win_size
        if self.use_hf:
            # tokens = torch.full((bsz, total_length), self.tokenizer.pad_token_id).long()
            tokens = torch.full((bsz, total_length), -1).long()
            bos_token = torch.full((bsz, 1), self.tokenizer.bos_token_id).long().cuda()
        else:
            params = self.model.params
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
            tokens = torch.full((bsz, total_length), self.tokenizer.pad_id).long()
            bos_token = torch.full((bsz, 1), self.tokenizer.bos_id).long().cuda()

        cumul = np.zeros(self.model.vocab_size+1, dtype = np.uint64)
        probs_list = []

        if starter_tokens is None:
            start_pos = 0
            prev_pos = -1
        else:
            tokens[:,:win_size]=torch.tensor(starter_tokens).long()
            start_pos = win_size
            prev_pos = 0
        tokens = tokens.cuda()
        
        # 新增：记录解码过程
        decoding_log = []

        for cur_pos in range(start_pos, total_length):
            if prev_pos == -1:
                logits = self.model.forward(bos_token, 0)
                prev_pos += 1
            elif cur_pos < win_size:
                logits = self.model.forward(torch.cat((bos_token,tokens[:, prev_pos:cur_pos]),1), 0)
            else:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], 0) #Position input to LLM is fixed to be 0
            
            probs = torch.softmax(logits, dim=-1) 
            probs_np = probs.cpu().numpy().reshape((-1,))

            probs_list += [probs_np]
            cumul[1:] = np.cumsum(probs_np*10000000 + 1)
            next_token = dec.read(cumul, probs_np.size)

            # 新增：保存原始算术解码结果
            if save_raw_results:
                self.raw_decoded_tokens.append(next_token)
                self.all_probability_distributions.append(probs_np.copy())
                
                # 记录解码详细信息
                decoding_log.append({
                    'position': cur_pos,
                    'token_id': int(next_token),
                    'token_text': self.tokenizer.decode([next_token]),
                    'probability': float(probs_np[next_token]),
                    'top5_tokens': self._get_top_tokens(probs_np, 5),
                    'top5_probs': self._get_top_probs(probs_np, 5)
                })

            tokens[:,cur_pos] = torch.tensor(next_token).long()
            if cur_pos >= win_size:
                prev_pos += 1

                # if (prev_pos*100/(total_length-win_size))%10 == 0:
                #     print(f'Decoder: Completed {int(prev_pos*100/(total_length-win_size))} %')
        
        # ---- 去掉 BOS <s> ----
        decoded_ids = tokens.tolist()[0]
        if self.use_hf:
            bos_id = self.tokenizer.bos_token_id
        else:
            bos_id = self.tokenizer.bos_id

        # 仅当开头的确是 BOS 时才去掉
        if len(decoded_ids) > 0 and decoded_ids[0] == bos_id:
            decoded_ids = decoded_ids[1:]

        decoded_text = self.tokenizer.decode(decoded_ids)
        # decoded_text = self.tokenizer.decode(tokens.tolist()[0])
            
        # 新增：保存原始解码结果到文件
        if save_raw_results:
            self._save_raw_results(decoding_log, raw_output_file)

        bitin.close()
        file_in.close()

        return decoded_text

    def _get_top_tokens(self, probs, top_k=5):
        """获取概率最高的top_k个token"""
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [self.tokenizer.decode([int(idx)]) for idx in top_indices]
    
    def _get_top_probs(self, probs, top_k=5):
        """获取概率最高的top_k个概率值"""
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [float(probs[idx]) for idx in top_indices]
    
    def _save_raw_results(self, decoding_log, output_file=None):
        """保存原始解码结果"""
        if output_file is None:
            output_file = 'arithmetic_decoding_raw_results.json'
        
        # 准备保存的数据
        save_data = {
            'raw_decoded_tokens': self.raw_decoded_tokens,
            'decoding_log': decoding_log,
            'total_tokens_decoded': len(self.raw_decoded_tokens),
            'vocab_size': self.model.vocab_size
        }
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存为CSV格式便于分析
        self._save_csv_results(decoding_log, output_file.replace('.json', '.csv'))
        
        print(f"原始算术解码结果已保存到: {output_file}")
    
    def _save_csv_results(self, decoding_log, csv_file):
        """保存CSV格式的结果"""
        import pandas as pd
        
        csv_data = []
        for log_entry in decoding_log:
            csv_data.append({
                'position': log_entry['position'],
                'token_id': log_entry['token_id'],
                'token_text': log_entry['token_text'],
                'probability': log_entry['probability'],
                'top1_token': log_entry['top5_tokens'][0],
                'top1_prob': log_entry['top5_probs'][0],
                'top2_token': log_entry['top5_tokens'][1],
                'top2_prob': log_entry['top5_probs'][1],
                'top3_token': log_entry['top5_tokens'][2],
                'top3_prob': log_entry['top5_probs'][2]
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV格式结果已保存到: {csv_file}")
    
    def get_raw_decoding_results(self):
        """获取原始算术解码结果"""
        return {
            'raw_tokens': self.raw_decoded_tokens,
            'probability_distributions': self.all_probability_distributions,
            'decoded_text': ''.join([self.tokenizer.decode([token]) for token in self.raw_decoded_tokens])
        }
        
    def decode_ranks(
        self,
        win_size,
        starter_tokens,
        compressed_file_name: str = 'LLMzip_RZ.txt'):

        with open(compressed_file_name,'rb') as file_in:
            ranks_compressed = file_in.read()
        
        ranks_decomp = zlib.decompress(ranks_compressed).decode('ascii')
        ranks_in = np.fromstring(ranks_decomp,sep=' ',dtype=np.int64)
        
        bsz = 1    # predicts 1 token at a time
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        total_length = ranks_in.shape[0]
        
        if starter_tokens is None:
            ranks = torch.tensor(ranks_in).reshape(bsz,-1).cuda()

        else:
            total_length += win_size

            ranks_in = np.append(np.zeros((win_size,),dtype=np.int64),ranks_in)
            ranks = torch.tensor(ranks_in).reshape(bsz,-1).cuda()
        if self.use_hf:
            bos_token = torch.full((bsz, 1), self.tokenizer.bos_token_id).long().cuda()
            # tokens = torch.full((bsz, total_length), self.tokenizer.pad_token_id).long()
            tokens = torch.full((bsz, total_length), -1).long()
        else:
            bos_token = torch.full((bsz, 1), self.tokenizer.bos_id).long().cuda()
            tokens = torch.full((bsz, total_length), self.tokenizer.pad_id).long()

        if starter_tokens is None:
            start_pos = 0
            prev_pos = -1
        else:
            tokens[:,:win_size]=torch.tensor(starter_tokens).long()
            start_pos = win_size
            prev_pos = 0
        tokens = tokens.cuda()

        for cur_pos in range(start_pos, total_length):
            if prev_pos == -1:
                logits = self.model.forward(bos_token, 0)
                prev_pos += 1
            elif cur_pos < win_size:
                logits = self.model.forward(torch.cat((bos_token,tokens[:, prev_pos:cur_pos]),1), 0)
            else:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], 0) #Position input to LLM is fixed to be 0
            
            probs = torch.softmax(logits, dim=-1) 
            next_token = gen_next_token(probs,ranks[:,cur_pos:cur_pos+1]) 
            tokens[:,cur_pos] = torch.tensor(next_token).long()

            if cur_pos >= win_size:
                prev_pos += 1

                # if (prev_pos*100/(total_length-win_size))%10 == 0:
                #     print(f'Decoder: Completed {int(prev_pos*100/(total_length-win_size))} %')

        decoded_text = self.tokenizer.decode(tokens.tolist()[0])

        return decoded_text