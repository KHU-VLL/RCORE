import json, torch, os
import torch.distributed as dist
import math
import torch
from torch import nn
from scipy.special import binom

def write_and_print(msg, log_file=None, end='\n'):
    if dist.get_rank() == 0:
        print(msg, end=end)
        if log_file:
            log_file.write(msg + end)
            log_file.flush()


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)
    v_o_on_v_o = v_o / (torch.sum(v_o, dim=(0, 1), keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o, v_o_on_v_o


def save_checkpoint(state, save_path, epoch, best=False, cur_epoch=-1):
    if dist.get_rank() != 0:
        return
    if cur_epoch != -1 :
        filename = os.path.join(save_path, f"epoch_resume_{cur_epoch}.pt")
    elif best :
        filename = os.path.join(save_path, f"best.pt")
    else :
        filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)
    print("save checkpoint : ", filename)