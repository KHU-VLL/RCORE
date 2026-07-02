import os
import pprint
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torch.multiprocessing
import torch.distributed as dist

from utils.opts import parser
from utils.loss import *

from engine.test_engine import predict_logits_ddp, test_bias, test_auc
from engine.utils import write_and_print
from dataset.com_video_dataset import CompositionVideoDataset
from train import (
    setup, cleanup, setup_for_distributed,
    set_seed, load_args,
    build_model, build_lora,
)


torch.multiprocessing.set_sharing_strategy('file_system')

STH_COM_PATH = "sth_data_path"
EK100_COM_PATH = "ek100_data_path"

EVAL_DATASET_OVERRIDES = {
    "sth-com": {
        "label_index_file":     "./data_split/sth_com/verb_label_dict.json",
        "obj_label_index_file": "./data_split/sth_com/object_label_dict.json",
        "dataset":              "sth-com",
        "split_root":           "./data_split/sth_com",
        "dataset_path":         STH_COM_PATH,
    },
    "ek100-com": {
        "label_index_file":     "./data_split/ek100-com/verb_label_dict.json",
        "obj_label_index_file": "./data_split/ek100-com/object_label_dict.json",
        "dataset":              "sth-com",
        "split_root":           "./data_split/ek100-com",
        "dataset_path":         EK100_COM_PATH,
    },
}


def resolve_config_path(config):
    return config.cfg_path if config.cfg_path else os.path.join(config.logpath, 'config.yml')


def maybe_override_for_eval(config):
    if config.dataset == config.eval_dataset:
        return
    overrides = EVAL_DATASET_OVERRIDES.get(config.eval_dataset)
    if overrides is None:
        return
    for k, v in overrides.items():
        setattr(config, k, v)


def build_test_datasets(config):
    return_ids = bool(getattr(config, 'save_features', False))
    common = dict(
        split='compositional-split-natural',
        tdn_input='tdn' in config.arch,
        frames_duration=config.num_frames,
        split_root=getattr(config, "split_root", None),
        open_world=True,
        return_ids=return_ids,
    )
    make = lambda phase: CompositionVideoDataset(config.dataset_path, phase=phase, **common)
    return make('train'), make('val'), make('test')


def _rename_ckpt_key(ckpt, key):
    """Strip `module.target.` / `module.` prefix in-place; return the new key."""
    for prefix in ('module.target.', 'module.'):
        if prefix in key:
            new_key = key.replace(prefix, "")
            ckpt[new_key] = ckpt.pop(key)
            return new_key
    return key


def load_test_checkpoint(model, config):
    ckpt_path = os.path.join(config.logpath, "best.pt")
    print("Model load : " + ckpt_path)
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']

    target = model.model.model if 'qwen' in config.method else model
    model_state = target.state_dict()

    for key in list(ckpt.keys()):
        if key.startswith("model.model.base_model.model.model."):
            del ckpt[key]
            continue

        key = _rename_ckpt_key(ckpt, key)

        if "prompt_learner" in key:
            if ckpt[key].shape != model_state[key].shape:
                print(f"Prompt learner shape mismatch: {ckpt[key].shape} vs {model_state[key].shape}. Skipping.")
                del ckpt[key]
                print("Also delete ctx, use a photo of")
                ctx_key = ".".join(key.split(".")[:-1]) + ".ctx"
                if ctx_key in ckpt:
                    del ckpt[ctx_key]
        if 'verb_prototypes' in key and key in ckpt:
            if ckpt[key].shape != model_state[key].shape:
                print(f"verb_prototypes shape mismatch: {ckpt[key].shape} vs {model_state[key].shape}. Skipping.")
                del ckpt[key]

    msg = target.load_state_dict(ckpt, strict=False)
    print("$$$$$$$$$$$$$")
    print(msg, flush=True)

    del ckpt
    torch.cuda.empty_cache()


def _seed_train_pairs_idx(*datasets):
    """Set `train_pairs_idx` on each dataset from the first dataset's train pairs + attr/obj vocab."""
    src = datasets[0]
    pairs_idx = [(src.attr2idx[a], src.obj2idx[o]) for a, o in src.train_pairs]
    for ds in datasets:
        ds.train_pairs_idx = pairs_idx


def build_pair_indices(test_dataset):
    """Return (open_pairs, open_pair2idx, open_idx2pair, closed_pair2idx, closed_idx2pair, closed_mask)."""
    open_pairs = list(product(test_dataset.attrs, test_dataset.objs))
    open_pair2idx = {pair: idx for idx, pair in enumerate(open_pairs)}
    open_idx2pair = {v: k for k, v in open_pair2idx.items()}
    print('Number of open pairs: %d' % len(open_pairs))

    closed_pairs = set(test_dataset.train_pairs + test_dataset.val_pairs + test_dataset.test_pairs)
    closed_pair2idx = {}
    for pair in open_pairs:
        if pair in closed_pairs:
            closed_pair2idx[pair] = len(closed_pair2idx)
    closed_idx2pair = {v: k for k, v in closed_pair2idx.items()}
    closed_mask = torch.BoolTensor([pair in closed_pairs for pair in open_pairs])
    print('Number of closed pairs: %d' % len(closed_pairs))
    return open_pairs, open_pair2idx, open_idx2pair, closed_pair2idx, closed_idx2pair, closed_mask


def build_gt_and_mask(all_attr_gt, all_obj_gt, idx2attr, idx2obj,
                     open_pair2idx, closed_pair2idx, train_pair_set):
    closed_comp_gt, open_comp_gt, seen_pair_mask = [], [], []
    for attr_label, obj_label in zip(all_attr_gt, all_obj_gt):
        pair = (idx2attr[attr_label.item()], idx2obj[obj_label.item()])
        closed_comp_gt.append(closed_pair2idx[pair])
        open_comp_gt.append(open_pair2idx[pair])
        seen_pair_mask.append(pair in train_pair_set)
    return (
        torch.Tensor(closed_comp_gt),
        torch.Tensor(open_comp_gt),
        torch.BoolTensor(seen_pair_mask),
    )


def main():
    config = parser.parse_args()

    config_path = resolve_config_path(config)
    print("CONFIG PATH : ", config_path)
    load_args(config_path, config)
    maybe_override_for_eval(config)

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(local_rank)
    torch.cuda.set_device(local_rank)

    log_file_path = None
    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(config.save_path, f'std_{timestamp}.out')
        os.makedirs(config.save_path, exist_ok=True)
    setup_for_distributed(local_rank == 0, log_file_path)

    print(config)
    set_seed(config.seed)
    print("Test details")
    pprint.pprint(config)

    train_dataset, val_dataset, test_dataset = build_test_datasets(config)
    config.attrs = train_dataset.attrs
    if hasattr(train_dataset, "objs"):
        config.objs = train_dataset.objs

    model = build_model(config).cuda(local_rank)
    if config.method in ("internvideo_c2c", "internvideo_c2c_1b"):
        model = build_lora(model, config)

    load_test_checkpoint(model, config)

    config.dataset = config.eval_dataset
    model.eval()
    if not os.path.exists(config.logpath):
        config.logpath = config.save_path
    log_test = open(
        os.path.join(config.logpath, f'test_log_{config.dataset}_{config.load_ckpt_type}.txt'),
        'w',
    )

    print("Evaluating test dataset:")

    if getattr(config, 'test_bias', False):
        print("### test_bias ###")
        _seed_train_pairs_idx(train_dataset, val_dataset, test_dataset)
        test_bias(model, val_dataset, test_dataset, config)
        print("done test_bias!")
        log_test.close()
        return

    if getattr(config, 'test_auc', False):
        print("### test_auc ###")
        _seed_train_pairs_idx(train_dataset, val_dataset, test_dataset)
        test_auc(model, test_dataset, config)
        print("done test_auc!")
        log_test.close()
        return

    print("PRED MODE : ", config.pred_mode)
    print("C2C_PRED MODE : ", config.c2c_pred_mode)

    assert dist.is_available() and dist.is_initialized(), \
        "distributed backend must be initialized to run predict_logits_ddp"
    all_logits, all_attr_gt, all_obj_gt, all_comp_gt, loss_avg = predict_logits_ddp(
        model, test_dataset, config,
    )

    (open_pairs, open_pair2idx, open_idx2pair,
     closed_pair2idx, closed_idx2pair, closed_mask) = build_pair_indices(test_dataset)

    idx2attr = {v: k for k, v in test_dataset.attr2idx.items()}
    idx2obj = {v: k for k, v in test_dataset.obj2idx.items()}
    attr2idx, obj2idx = test_dataset.attr2idx, test_dataset.obj2idx

    closed_comp_gt, open_comp_gt, seen_pair_mask = build_gt_and_mask(
        all_attr_gt, all_obj_gt, idx2attr, idx2obj,
        open_pair2idx, closed_pair2idx,
        train_pair_set=set(train_dataset.train_pairs),
    )
    print('Number of seen pair samples: %d' % seen_pair_mask.sum().item())

    def get_accuracies(logits, idx2pair, comp_gt):
        top1_comp_pred = logits.argmax(dim=1)
        top1_comp_pred_name = [idx2pair[c.item()] for c in top1_comp_pred]
        top1_verb_pred = torch.Tensor([attr2idx[v] for v, _ in top1_comp_pred_name])
        top1_obj_pred = torch.Tensor([obj2idx[o] for _, o in top1_comp_pred_name])

        def _acc(pred, gt, mask):
            return (pred[mask] == gt[mask]).float().mean().item()

        seen = seen_pair_mask
        unseen = ~seen_pair_mask
        assert len(top1_comp_pred[seen]) + len(top1_comp_pred[unseen]) == len(all_logits)
        return (
            _acc(top1_comp_pred, comp_gt,     seen), _acc(top1_verb_pred, all_attr_gt, seen), _acc(top1_obj_pred, all_obj_gt, seen),
            _acc(top1_comp_pred, comp_gt,     unseen), _acc(top1_verb_pred, all_attr_gt, unseen), _acc(top1_obj_pred, all_obj_gt, unseen),
        )

    def get_macro_cg(logits, idx2pair, comp_gt):
        top1_comp_pred = logits.argmax(dim=1)
        top1_comp_pred_name = [idx2pair[c.item()] for c in top1_comp_pred]
        top1_verb_pred = torch.tensor([attr2idx[v] for v, _ in top1_comp_pred_name], device=comp_gt.device)
        top1_obj_pred = torch.tensor([obj2idx[o] for _, o in top1_comp_pred_name], device=comp_gt.device)

        def _macro(mask):
            mc_pred, mc_gt = top1_comp_pred[mask], comp_gt[mask]
            mv_pred, mv_gt = top1_verb_pred[mask], all_attr_gt[mask]
            mo_pred, mo_gt = top1_obj_pred[mask], all_obj_gt[mask]

            unique_classes = torch.unique(mc_gt)
            if len(unique_classes) == 0:
                return 0.0

            scores = []
            for c in unique_classes:
                class_mask = (mc_gt == c)
                comp_acc = (mc_pred[class_mask] == mc_gt[class_mask]).float().mean().item() * 100
                verb_acc = (mv_pred[class_mask] == mv_gt[class_mask]).float().mean().item() * 100
                obj_acc = (mo_pred[class_mask] == mo_gt[class_mask]).float().mean().item()
                scores.append(comp_acc - (verb_acc * obj_acc))
            print(len(scores))
            return sum(scores) / len(scores)

        assert len(top1_comp_pred[seen_pair_mask]) + len(top1_comp_pred[~seen_pair_mask]) == len(all_logits)
        return _macro(seen_pair_mask), _macro(~seen_pair_mask)

    def wp(msg):
        write_and_print(msg, log_test)

    # --- Open world ---
    (seen_comp_acc, seen_verb_acc, seen_obj_acc,
     unseen_comp_acc, unseen_verb_acc, unseen_obj_acc) = get_accuracies(
        all_logits, open_idx2pair, open_comp_gt,
    )
    seen_macro_cg, unseen_macro_cg = get_macro_cg(all_logits, open_idx2pair, open_comp_gt)

    wp("=== Open World ===")
    wp("[Seen]")
    wp(f"Top-1 Comp Acc: {seen_comp_acc:.4f}")
    wp(f"Top-1 Verb Acc: {seen_verb_acc:.4f}")
    wp(f"Top-1 Obj Acc: {seen_obj_acc:.4f}")
    wp(f"Macro CG: {seen_macro_cg:.4f}")
    wp(f"Original CG: {seen_comp_acc * 100 - seen_verb_acc * seen_obj_acc * 100:.4f}")
    wp("\n[Uneen]")
    wp(f"Top-1 Comp Acc: {unseen_comp_acc:.4f}")
    wp(f"Top-1 Verb Acc: {unseen_verb_acc:.4f}")
    wp(f"Top-1 Obj Acc: {unseen_obj_acc:.4f}")
    wp(f"Macro CG: {unseen_macro_cg:.4f}")
    wp(f"Original CG: {unseen_comp_acc * 100 - unseen_verb_acc * unseen_obj_acc * 100:.4f}")
    wp("\n[HM]")
    wp(f"Verb HM: {2 / (1 / seen_verb_acc + 1 / unseen_verb_acc):.4f}")
    wp(f"Obj HM: {2 / (1 / seen_obj_acc + 1 / unseen_obj_acc):.4f}")
    wp(f"Comp HM: {2 / (1 / seen_comp_acc + 1 / unseen_comp_acc):.4f}")

    # --- Closed world ---
    (seen_comp_acc, seen_verb_acc, seen_obj_acc,
     unseen_comp_acc, unseen_verb_acc, unseen_obj_acc) = get_accuracies(
        all_logits[:, closed_mask], closed_idx2pair, closed_comp_gt,
    )

    wp("\n\n=== Closed World ===")
    wp("[Seen]")
    wp(f"Top-1 Comp Acc: {seen_comp_acc:.4f}")
    wp(f"Top-1 Verb Acc: {seen_verb_acc:.4f}")
    wp(f"Top-1 Obj Acc: {seen_obj_acc:.4f}")
    wp("\n[Uneen]")
    wp(f"Top-1 Comp Acc: {unseen_comp_acc:.4f}")
    wp(f"Top-1 Verb Acc: {unseen_verb_acc:.4f}")
    wp(f"Top-1 Obj Acc: {unseen_obj_acc:.4f}")

    if getattr(config, 'save_tensors', False):
        pred_mode = getattr(config, 'pred_mode', "single")
        torch.save(all_logits, os.path.join(config.logpath, 'all_logits.pt'))
        torch.save(all_attr_gt, os.path.join(config.logpath, 'all_attr_gt.pt'))
        if pred_mode == "both":
            torch.save(all_obj_gt, os.path.join(config.logpath, 'all_obj_gt.pt'))

    log_test.close()
    print("done!")


if __name__ == "__main__":
    main()
    exit(0)
