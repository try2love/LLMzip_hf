from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLMzip_encode, LLMzip_decode
from llama.llmzip_utils import parse_win_nt_from_filename
### Command to run
# torchrun --nproc_per_node 1 LLMzip_run.py --ckpt_dir weights/7B --tokenizer_path weights/tokenizer.model
# --win_len 511 --text_file *.txt --compression_folder LLMzip_compression   > Log_files/text8_ent1.txt 2>&1

### For precise reproduction of the paper results set the following options
# compression_alg - 'both', encode_decode - 0, batched_encode = True, verify_save_decoded = 0, with_context_start = True

class DummyParams:
    def __init__(self, max_seq_len, max_batch_size):
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

def setup_model_parallel() -> Tuple[int, int]:
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    # 非分布式场景
    if local_rank_env is None or world_size_env is None:
        print("No torchrun env detected: running single-process, single-GPU.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        torch.manual_seed(1)
        return 0, 1

    # 分布式场景
    local_rank = int(local_rank_env)
    world_size = int(world_size_env)
    print("Local Rank : ", local_rank, ", World Size : ", world_size)

    torch.distributed.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    lora_dir: str = None,
):
    start_time = time.time()
    # If the checkpoint directory looks like a HuggingFace model (config.json exists),
    # use the HF adapter so the script can accept HF-format llama-2-7b-hf directories.
    hf_config_path = os.path.join(ckpt_dir, "config.json")
    # print(hf_config_path)
    
    if os.path.exists(hf_config_path):
        try:
            from llama.hf_adapter import HFTransformerAdapter
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        except Exception as e:
            raise RuntimeError("HF adapter import failed. Make sure llama/hf_adapter.py exists and transformers is installed.") from e

        print("Detected HuggingFace-format model in ckpt_dir. Using HF adapter to load model.")

        model_adapter = HFTransformerAdapter(hf_model_dir=ckpt_dir,lora_dir=lora_dir)
        # expose attribute expected by LLMzip (vocab_size)
        # model_adapter.vocab_size is already set in adapter
        Encoder = LLMzip_encode(model_adapter, tokenizer,True)
        Decoder = LLMzip_decode(model_adapter, tokenizer,True)
        print(f"Loaded HF model in {time.time() - start_time:.2f} seconds")
        return Encoder, Decoder, True

    tokenizer = Tokenizer(model_path=tokenizer_path)
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # 确保检查点数量与分布式训练的进程数匹配
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # 每个进程加载自己对应的检查点分片（通过 local_rank 索引）
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    # 使用 map_location="cpu" 先加载到 CPU，避免 GPU 内存不足
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    Encoder = LLMzip_encode(model, tokenizer,False)
    Decoder = LLMzip_decode(model, tokenizer,False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return Encoder,Decoder,False


def verify_text(compressed_file_name,text_file,text_decoded,context_txt,save_decoded,alg):
    with open(text_file,'r') as txt_enc:
        text_encoded = txt_enc.read()

    if context_txt is not None:
        text_encoded = text_encoded[len(context_txt):]
        text_decoded = text_decoded[len(context_txt):]

    if text_encoded == text_decoded:
        print(f'Successful decoding using {alg}')
    else:
        print("********!!!!! Error !!!!!*********")
        print("***********Encoded Text************")
        print(text_encoded)
        print("***********Decoded Text************")
        print(text_decoded)

    if save_decoded:
        if alg == 'ArithmeticCoding':
            with open(compressed_file_name+'_AC_decoded_text.txt','w') as txt_dec:
                txt_dec.write(text_decoded)
        else:
            with open(compressed_file_name+'_RZ_decoded_text.txt','w') as txt_dec:
                txt_dec.write(text_decoded )

import os

def get_file_size_bits(file_path):
    """
    计算文件的真实占用比特数
    file_path (str): 文件路径
    Returns: int: 文件的比特数
    """
    try:
        # 获取文件大小（字节数）
        file_size_bytes = os.path.getsize(file_path)
        # 转换为比特数
        file_size_bits = file_size_bytes * 8
        return file_size_bits
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except Exception as e:
        raise Exception(f"计算文件大小时出错: {e}")

def calculate_compression_ratio(original_text_file: str, compressed_file_path: str) -> float:
    """Calculate semantic compression ratio (p1)"""
    b_o = get_file_size_bits(original_text_file)
    compressed_size = get_file_size_bits(compressed_file_path)
    print(f"原始文件比特数：{b_o}；压缩文件比特数：{compressed_size}")
    return (b_o - compressed_size) / b_o

def calculate_recovery_rate(original_text: str, decoded_text: str) -> float:
    """Calculate information recovery rate (p2)"""
    set_a = set(original_text)
    set_b = set(decoded_text)
    intersection = set_a & set_b
    return len(intersection) / len(set_a)

local_rank, world_size = setup_model_parallel()
print(f"local_rank = {local_rank}, world_size={world_size}")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    win_len: int,
    text_file: str,
    compression_folder: str,
    compressed_file_name:str,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    compression_alg: str = 'ArithmeticCoding',
    encode_decode: int = 2,
    batched_encode: bool = False,
    verify_save_decoded: int = 2,
    with_context_start: bool = False,
    self_calculate_p: bool = False,
    lora_dir: str = None
):

    # win_len - The context window length and it cannot exceed the max seq length 512 上下文长度
    # compression_alg -  ArithmeticCoding / RankZip / both 压缩算法选择
    # encode_decode - 0: Only encode, 1: only decode, 2: both 压缩or解压算法
    # batched_encode - Use only for faster encoding (theoretical entropy computations), 
    #                  decoding doesn't work with batched encoding
    # with_context_start - avoids encoding the initial context , and provides the initial context at the decoder
    # verify_save_decoded - 0: don't verify/save, 1: only verify, 2: verify and save
    # Specify in_file with extension and out_file_name without extension
    # 初始保留的上下文长度必须不大于最大长度
    assert win_len <= max_seq_len, f'Window length {win_len} is greater than {max_seq_len}'
    assert encode_decode in [0,1,2], f'encode_decode not in {[0,1,2]}'
    assert compression_alg in ['ArithmeticCoding','RankZip','both'], 'compression_alg not one of ArithmeticCoding / RankZip / both'

    if batched_encode:
        print("Warning decoding doesn't work when using batched encode")

    start_time_main = time.time()

    # 当前进程非主进程，则不打印日志和输出信息
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    
    encode = encode_decode%2 == 0 # Convert to Bool
    decode = encode_decode>0      # Convert to Bool
    
    if decode:
        batched_encode = False
    use_hf = True
    Encoder,Decoder,use_hf = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size, lora_dir)

    os.makedirs(compression_folder,exist_ok=True)
    # compressed_file_name = compression_folder + f'/LLMzip_{win_len}' 

    with open(text_file,'r') as f_in:
            text_input = f_in.read()

    if encode:
        # Only encoding
        # 将文本转换为token序列
        if use_hf:
            tokens_full = np.array(Encoder.tokenizer.encode(text_input))
        else:
            tokens_full = np.array(Encoder.tokenizer.encode(text_input,bos=False,eos=False))
        # print(f"tokens:\n{tokens_full}")

        # 处理上下文起始标记（可选
        if with_context_start:
            # 提取前win_len个token作为起始上下文
            starter_tokens = tokens_full[:win_len]
            # 保存起始上下文token
            np.save(compressed_file_name+'_starter_tokens.npy',starter_tokens)

        # If the same tokens need to be encoded for any win_len (This has been used for our work)
        # tokens_full = np.array(Encoder.tokenizer.encode(text_input,bos=False,eos=False))[511-win_len:]
        # 调用编码器对token序列进行压缩编码
        # 参数说明：
        # - win_len: 滑动窗口大小
        # - compression_alg: 压缩算法类型
        # - compressed_file_name: 压缩文件保存路径
        # - tokens_full: 完整token序列
        # - batched_encode: 是否使用批处理编码
        # - with_context_start: 是否包含上下文起始标记
        N_C, N_T, Bits, compressed_file_name_full = Encoder.encode_from_tokens(win_len,compression_alg,compressed_file_name,tokens_full=tokens_full,batched_encode=batched_encode,with_context_start=with_context_start,out_dir=compression_folder)
    # 解码逻辑变更为关键参数从pkl文件中读取
    if decode:
        # with open(compressed_file_name+'_metrics.json') as metrics_file:
            # total_length = json.load(metrics_file)['$N_T$'][0] #Load number of tokens from compression metrics for arithmetic coding length
        compressed_file_name_full = next((p for p in Path().rglob(f"{compressed_file_name}*.pkl") if p.is_file()), None)
        win_len, total_length = parse_win_nt_from_filename(compressed_file_name_full)
        if with_context_start:
            starter_tokens = np.load(compressed_file_name+'_starter_tokens.npy')
            context_txt = Encoder.tokenizer.decode(starter_tokens.tolist())
        else:
            starter_tokens = None
            context_txt = None

        if (compression_alg == 'ArithmeticCoding')or(compression_alg =='both'): 
            # compressed_file_name_full = compressed_file_name+'_AC.txt'
            
            decoded_text_ac = Decoder.decode_AC(win_len,starter_tokens,total_length, compressed_file_name_full, save_raw_results=False)
            if verify_save_decoded > 0:
                verify_text(compressed_file_name,text_file,decoded_text_ac,context_txt,verify_save_decoded==2,'ArithmeticCoding')
            
        if (compression_alg == 'RankZip')or(compression_alg =='both'): 
            # compressed_file_name_full = compressed_file_name+'_RZ.txt'
            decompressed_file_name = compressed_file_name+'_RZ'

            decoded_text_rz = Decoder.decode_ranks(win_len,starter_tokens, compressed_file_name_full)
            if verify_save_decoded > 0:
                verify_text(compressed_file_name,text_file,decoded_text_rz,context_txt,verify_save_decoded==2,'RankZip')

    print(f"Completed in {time.time() - start_time_main:.2f} seconds")
    if self_calculate_p:
        if encode_decode == 0: # encode only
            p1 = calculate_compression_ratio(text_file, compressed_file_name_full)
            p2 = 1.0
            print(f"只进行文本压缩，语义压缩比：{p1}")
        elif encode_decode == 1: #decode only
            if compression_alg == 'ArithmeticCoding' or compression_alg == 'both':
                with open(f"{compressed_file_name}_AC_decoded_text.txt", 'r', encoding='utf-8') as dec_f:
                    decoded_text = dec_f.read()
            if (compression_alg == 'RankZip')or(compression_alg =='both'):
                with open(f"{compressed_file_name}_RZ_decoded_text.txt", 'r', encoding='utf-8') as dec_f:
                    decoded_text = dec_f.read()
            p1 = 1.0
            p2 = 1.0
            print(f"只进行文本恢复，数值均无法计算。恢复结果：\n{decoded_text}")
        else: # encode-decode
            if compression_alg == 'ArithmeticCoding' or compression_alg == 'both':
                with open(f"{compressed_file_name}_AC_decoded_text.txt", 'r', encoding='utf-8') as dec_f:
                    decoded_text = dec_f.read()
            if (compression_alg == 'RankZip')or(compression_alg =='both'):
                with open(f"{compressed_file_name}_RZ_decoded_text.txt", 'r', encoding='utf-8') as dec_f:
                    decoded_text = dec_f.read()
            p1 = calculate_compression_ratio(text_file, compressed_file_name_full)
            p2 = calculate_recovery_rate(text_input, decoded_text)
            print(f"语义压缩比：{p1}\n信息恢复率：{p2}")
        return p1,p2,N_C, N_T, Bits
if __name__ == "__main__":
    fire.Fire(main)
