import torch
import numpy as np
import pickle
import re
import os

def build_compressed_pkl_name(input_filepath: str, out_dir: str, win_len: int, N_T: int, prefix: str = None):
    """
    Construct output filename: {prefix_or_inputbasename}_win{win_len}_NT{N_T}.pkl
    """
    if prefix:
        base = prefix
    else:
        base = os.path.splitext(os.path.basename(input_filepath))[0]
    fname = f"{base}_win{win_len}_NT{N_T}.pkl"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, fname)
    else:
        return fname

def save_bitstream_to_pkl(bitstream, output_path: str):
    """
    Save only the bitstream to a .pkl file.
    bitstream should be bytes or a Python object representing encoded bits.
    """
    with open(output_path, "wb") as f:
        pickle.dump(bitstream, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bitstream_from_pkl(pkl_path: str):
    """Load bitstream object from pkl"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def parse_win_nt_from_filename(filename: str):
    """
    Parse win_len and N_T from filename pattern *_win{win}_NT{NT}.pkl
    Returns (win_len:int or None, N_T:int or None)
    """
    b = os.path.basename(filename)
    m = re.search(r"_win(\d+)_NT(\d+)\.pkl$", b)
    if m:
        return int(m.group(1)), int(m.group(2))
    # try more permissive pattern (maybe extension .ac.pkl etc)
    m2 = re.search(r"_win(\d+)_NT(\d+)", b)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return None, None

def gen_rank(probs,next_token):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True) 
    rank_list = []
    if next_token.shape[0]>1:
        for i in range(next_token.shape[0]):
            rank_list += [torch.where(probs_idx[i:i+1,:] == next_token[i])[-1]]
        rank = torch.squeeze(torch.stack(rank_list))
    else:
        rank = torch.where(probs_idx == next_token)[-1]
    return rank

def gen_next_token(probs,rank):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True)
    next_token = torch.gather(probs_idx, -1, rank)
    return next_token

def read_bitstream(bitin):
    temp_list = []
    while True:
        temp = bitin.read()
        if temp == -1:
            break
        temp_list += [temp]
    temp_arr = np.array(temp_list)
    final_ind = (np.where(temp_arr==1)[0][-1]).astype(int)
    final_arr = temp_arr[:final_ind+1]
    
    return final_arr

def get_str_array(array):
    array_used = array.reshape(-1)
    str_out = str()
    for i in range(array_used.size):
        str_out +=str(array_used[i])+" "
    return str_out



