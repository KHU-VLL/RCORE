import os
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import *

cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _get_module(model):
    return model.module if hasattr(model, "module") else model


def _build_ddp_dataloader(dataset, config, collate_fn=None):
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def _autocast(config, default_dtype=torch.float32):
    dtype_ = torch.float16 if getattr(config, "use_flash_attn", False) else default_dtype
    return torch.amp.autocast(device_type='cuda', enabled=True, dtype=dtype_)


def gather_across_gpus(tensor):
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def _build_pair_indices(dataset):
    """Return (open_pairs, open_idx2pair_ids, open_idx2pair_names, seen_mask, closed_mask)."""
    open_pairs = list(product(dataset.attrs, dataset.objs))
    open_idx2pair_ids = {
        idx: (dataset.attr2idx[a], dataset.obj2idx[o])
        for idx, (a, o) in enumerate(open_pairs)
    }
    open_idx2pair_names = {idx: pair for idx, pair in enumerate(open_pairs)}
    seen_mask = torch.BoolTensor([pair in dataset.train_pairs for pair in open_pairs])
    total_pairs = dataset.train_pairs + dataset.val_pairs + dataset.test_pairs
    closed_mask = torch.BoolTensor([pair in total_pairs for pair in open_pairs])
    return open_pairs, open_idx2pair_ids, open_idx2pair_names, seen_mask, closed_mask


def _forward_com_logits(model, batch_img):
    """Model returns single com_logits when save_features=False, else (verb, obj, com)."""
    output = model(batch_img)
    if isinstance(output, torch.Tensor):
        return output
    return output[2]


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #


def get_macro_cg(logits, idx2pair, comp_gt, all_attr_gt, all_obj_gt, seen_pair_mask,
                 attr2idx, obj2idx):
    def _custom_metric(c_pred, v_pred, o_pred, c_gt, v_gt, o_gt, mask):
        mc_pred, mc_gt = c_pred[mask], c_gt[mask]
        mv_pred, mv_gt = v_pred[mask], v_gt[mask]
        mo_pred, mo_gt = o_pred[mask], o_gt[mask]

        unique_classes = torch.unique(mc_gt)
        if len(unique_classes) == 0:
            return 0.0

        scores = []
        for c in unique_classes:
            class_mask = (mc_gt == c)
            comp_acc = (mc_pred[class_mask] == mc_gt[class_mask]).float().mean().item() * 100
            verb_acc = (mv_pred[class_mask] == mv_gt[class_mask]).float().mean().item() * 100
            # NOTE: `obj_acc` intentionally not scaled by 100 — preserved from original.
            obj_acc = (mo_pred[class_mask] == mo_gt[class_mask]).float().mean().item()
            scores.append(comp_acc - (verb_acc * obj_acc))
        return sum(scores) / len(scores)

    top1_comp_pred = logits.argmax(dim=1)
    top1_pairs = [idx2pair[c.item()] for c in top1_comp_pred]
    top1_verb_pred = torch.tensor([attr2idx[v] for v, _ in top1_pairs], device=comp_gt.device)
    top1_obj_pred = torch.tensor([obj2idx[o] for _, o in top1_pairs], device=comp_gt.device)

    seen_score = _custom_metric(
        top1_comp_pred, top1_verb_pred, top1_obj_pred,
        comp_gt, all_attr_gt, all_obj_gt, seen_pair_mask,
    )
    unseen_score = _custom_metric(
        top1_comp_pred, top1_verb_pred, top1_obj_pred,
        comp_gt, all_attr_gt, all_obj_gt, ~seen_pair_mask,
    )

    assert len(top1_comp_pred[seen_pair_mask]) + len(top1_comp_pred[~seen_pair_mask]) == len(logits)
    return seen_score, unseen_score


def draw_seen_unseen_curve(biases, val_results, test_results, best_results, save_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    val_hms, val_seen_accs, val_unseen_accs = val_results
    test_hm, test_seen_acc, test_unseen_acc = test_results
    best_bias, best_hm = best_results

    plt.figure(figsize=(12, 8))
    p_hm, = plt.plot(biases, val_hms, label='Validation HM', color='black', linewidth=2.5)
    p_seen, = plt.plot(biases, val_seen_accs, label='Validation Seen Acc', linestyle='--', alpha=0.7)
    p_unseen, = plt.plot(biases, val_unseen_accs, label='Validation Unseen Acc', linestyle='--', alpha=0.7)

    plt.axvline(x=best_bias, color='red', linestyle=':',
                label=f'Best Bias: {best_bias:.4f} (Val HM: {best_hm*100:.2f}%)')
    plt.plot(best_bias, best_hm, 'o', color='red', markersize=8)

    plt.scatter([best_bias], [test_hm], color=p_hm.get_color(),
                s=200, label=f'Test HM: {test_hm*100:.2f}%',
                marker='*', zorder=5, edgecolors='white')
    plt.scatter([best_bias], [test_seen_acc], color=p_seen.get_color(),
                s=200, label=f'Test Seen Acc: {test_seen_acc*100:.2f}%',
                marker='*', zorder=5, edgecolors='white')
    plt.scatter([best_bias], [test_unseen_acc], color=p_unseen.get_color(),
                s=200, label=f'Test Unseen Acc: {test_unseen_acc*100:.2f}%',
                marker='*', zorder=5, edgecolors='white')

    plt.title('Bias Calibration Curve and Final Test Performance', fontsize=16)
    plt.xlabel('Bias Value', fontsize=12)
    plt.ylabel('Accuracy / Harmonic Mean', fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(bottom=0)
    plt.xlim(biases.min(), biases.max())
    plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bias_curve.png'), dpi=300)
    print("\nPlot save !")


def get_accs(logits, idx2pair, all_gts, is_seen_sample):
    is_unseen_sample = ~is_seen_sample
    all_attr_gt, all_obj_gt, all_comp_gt = all_gts
    device = all_comp_gt.device

    predictions = torch.argmax(logits, dim=1)
    predicted_pairs = [idx2pair[p.item()] for p in predictions]
    predicted_attrs = torch.tensor([p[0] for p in predicted_pairs], device=device)
    predicted_objs = torch.tensor([p[1] for p in predicted_pairs], device=device)

    # Top-3
    attr_map = torch.tensor([idx2pair[i][0] for i in range(len(idx2pair))], device=device)
    obj_map = torch.tensor([idx2pair[i][1] for i in range(len(idx2pair))], device=device)
    _, top3_preds = torch.topk(logits, k=3, dim=1)
    top3_attrs = attr_map[top3_preds]
    top3_objs = obj_map[top3_preds]

    def _metrics(mask):
        if mask.sum() == 0:
            return [0.0] * 6
        n = mask.sum().item()
        acc = (predictions[mask] == all_comp_gt[mask]).sum().item() / n
        v_acc = (predicted_attrs[mask] == all_attr_gt[mask]).sum().item() / n
        o_acc = (predicted_objs[mask] == all_obj_gt[mask]).sum().item() / n
        acc3 = (top3_preds[mask] == all_comp_gt[mask].unsqueeze(1)).any(dim=1).sum().item() / n
        v_acc3 = (top3_attrs[mask] == all_attr_gt[mask].unsqueeze(1)).any(dim=1).sum().item() / n
        o_acc3 = (top3_objs[mask] == all_obj_gt[mask].unsqueeze(1)).any(dim=1).sum().item() / n
        return acc, v_acc, o_acc, acc3, v_acc3, o_acc3

    (seen_acc, seen_verb_acc, seen_obj_acc,
     seen_acc3, seen_verb_acc3, seen_obj_acc3) = _metrics(is_seen_sample)
    (unseen_acc, unseen_verb_acc, unseen_obj_acc,
     unseen_acc3, unseen_verb_acc3, unseen_obj_acc3) = _metrics(is_unseen_sample)

    current_hm = ((2 * seen_acc * unseen_acc) / (seen_acc + unseen_acc)
                  if (seen_acc + unseen_acc) > 0 else 0)

    def _cg(comp_acc, verb_acc, obj_acc):
        return comp_acc * 100 - (verb_acc * obj_acc * 100)

    print("-" * 71)
    print(f"Test Harmonic Mean (HM): {current_hm * 100:.2f}%")
    print(f"Seen   | Top-1: {seen_acc*100:5.2f}% (CG : {_cg(seen_acc, seen_verb_acc, seen_obj_acc):.2f}) (V: {seen_verb_acc*100:5.2f}%, O: {seen_obj_acc*100:5.2f}%)")
    print(f"       | Top-3: {seen_acc3*100:5.2f}% (CG : {_cg(seen_acc3, seen_verb_acc3, seen_obj_acc3):.2f})(V: {seen_verb_acc3*100:5.2f}%, O: {seen_obj_acc3*100:5.2f}%)")
    print(f"Unseen | Top-1: {unseen_acc*100:5.2f}% (CG : {_cg(unseen_acc, unseen_verb_acc, unseen_obj_acc):.2f}) (V: {unseen_verb_acc*100:5.2f}%, O: {unseen_obj_acc*100:5.2f}%)")
    print(f"       | Top-3: {unseen_acc3*100:5.2f}% (CG : {_cg(unseen_acc3, unseen_verb_acc3, unseen_obj_acc3):.2f})(V: {unseen_verb_acc3*100:5.2f}%, O: {unseen_obj_acc3*100:5.2f}%)")
    print("-" * 71)

    return current_hm, seen_acc, unseen_acc


# --------------------------------------------------------------------------- #
# Bias calibration
# --------------------------------------------------------------------------- #


@dataclass
class BiasCalibrationResult:
    hm: float = 0.0
    bias: float = 0.0
    seen_acc: float = 0.0
    unseen_acc: float = 0.0
    seen_verb_acc: float = 0.0
    seen_obj_acc: float = 0.0
    unseen_verb_acc: float = 0.0
    unseen_obj_acc: float = 0.0


def _bias_grid_search(base_logits, negative_mask, gts, idx2pair, bias_candidates, is_seen_sample):
    """Sweep `bias_candidates`, adding to `logits[:, negative_mask]`.

    Returns (best_result, seen_accs, unseen_accs, hms).
    """
    all_attr_gt, all_obj_gt, all_comp_gt = gts
    device = all_comp_gt.device
    is_unseen_sample = ~is_seen_sample
    num_seen = is_seen_sample.sum().item()
    num_unseen = is_unseen_sample.sum().item()

    best = BiasCalibrationResult()
    seen_accs, unseen_accs, hms = [], [], []

    print(f"Bias Calibration : {bias_candidates.min()} to {bias_candidates.max()}")
    for bias in tqdm(bias_candidates, desc="Calibrating Bias"):
        biased_logits = base_logits.clone()
        biased_logits[:, negative_mask] = biased_logits[:, negative_mask] + float(bias)

        predictions = torch.argmax(biased_logits, dim=1)
        predicted_pairs = [idx2pair[p.item()] for p in predictions]
        predicted_attrs = torch.tensor([p[0] for p in predicted_pairs], device=device)
        predicted_objs = torch.tensor([p[1] for p in predicted_pairs], device=device)

        seen_acc = seen_verb_acc = seen_obj_acc = 0.0
        unseen_acc = unseen_verb_acc = unseen_obj_acc = 0.0
        if num_seen > 0:
            seen_acc = (predictions[is_seen_sample] == all_comp_gt[is_seen_sample]).sum().item() / num_seen
            seen_verb_acc = (predicted_attrs[is_seen_sample] == all_attr_gt[is_seen_sample]).sum().item() / num_seen
            seen_obj_acc = (predicted_objs[is_seen_sample] == all_obj_gt[is_seen_sample]).sum().item() / num_seen
        if num_unseen > 0:
            unseen_acc = (predictions[is_unseen_sample] == all_comp_gt[is_unseen_sample]).sum().item() / num_unseen
            unseen_verb_acc = (predicted_attrs[is_unseen_sample] == all_attr_gt[is_unseen_sample]).sum().item() / num_unseen
            unseen_obj_acc = (predicted_objs[is_unseen_sample] == all_obj_gt[is_unseen_sample]).sum().item() / num_unseen

        current_hm = ((2 * seen_acc * unseen_acc) / (seen_acc + unseen_acc)
                      if seen_acc + unseen_acc > 0 else 0)

        seen_accs.append(seen_acc)
        unseen_accs.append(unseen_acc)
        hms.append(current_hm)

        if current_hm > best.hm:
            print(f"find best! Bias: {bias}, prev best seen/unseen: {best.seen_acc:.4f}/{best.unseen_acc:.4f}")
            best = BiasCalibrationResult(
                hm=current_hm, bias=float(bias),
                seen_acc=seen_acc, unseen_acc=unseen_acc,
                seen_verb_acc=seen_verb_acc, seen_obj_acc=seen_obj_acc,
                unseen_verb_acc=unseen_verb_acc, unseen_obj_acc=unseen_obj_acc,
            )

    return best, np.array(seen_accs), np.array(unseen_accs), np.array(hms)


# --------------------------------------------------------------------------- #
# Eval loops
# --------------------------------------------------------------------------- #


def _collect_ow_logits(model, dataloader, config):
    """Gather com_logits + labels across GPUs (compositional dataset)."""
    all_c_logits_chunks = []
    all_attr_gt, all_obj_gt, all_comp_gt = [], [], []
    total_count = 0

    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            batch_img = data[0].cuda(non_blocking=True)
            batch_attr = data[1].cuda(non_blocking=True)
            batch_obj = data[2].cuda(non_blocking=True)
            batch_target = data[3].cuda(non_blocking=True)

            with _autocast(config):
                com_logits = _forward_com_logits(model, batch_img)

            all_c_logits_chunks.append(gather_across_gpus(com_logits).cpu())
            all_attr_gt.append(gather_across_gpus(batch_attr).cpu())
            all_obj_gt.append(gather_across_gpus(batch_obj).cpu())
            all_comp_gt.append(gather_across_gpus(batch_target).cpu())

            count = torch.tensor(batch_img.size(0), device=batch_img.device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            total_count += count.item()

    return (torch.cat(all_c_logits_chunks, dim=0),
            torch.cat(all_attr_gt),
            torch.cat(all_obj_gt),
            torch.cat(all_comp_gt),
            total_count)


def predict_logits_ddp(model, dataset, config, cur_epo=0, collate_fn=None):
    model.eval()
    dataloader = _build_ddp_dataloader(dataset, config, collate_fn)

    pred_mode = getattr(config, 'pred_mode', "single")
    pred_single_mode = getattr(config, 'pred_single_mode', "verb")

    all_logits_chunks = []
    all_attr_gt, all_obj_gt, all_comp_gt = [], [], []
    loss_fn = CrossEntropyLoss()
    loss_sum, total_count, correct_count = 0.0, 0, 0

    print(f"Pred mode : {pred_mode}, Pred single mode : {pred_single_mode}")

    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            batch_img = data[0]
            if isinstance(batch_img, torch.Tensor):
                batch_img = batch_img.cuda(non_blocking=True)
            elif isinstance(batch_img, dict):
                batch_img = {k: v.cuda(non_blocking=True) for k, v in batch_img.items()}

            if pred_mode == "single":
                src = data[1] if pred_single_mode == 'verb' else data[2]
                batch_attr = src.cuda(non_blocking=True)
                batch_target = batch_attr  # single mode uses attr as the label
            else:  # "both"
                batch_attr = data[1].cuda(non_blocking=True)
                batch_obj = data[2].cuda(non_blocking=True)
                batch_target = data[3].cuda(non_blocking=True)

            with _autocast(config):
                pred = model(batch_img)

            loss = loss_fn(pred, batch_target)
            _, predicted = torch.max(pred, 1)
            batch_correct = (predicted == batch_target).sum()

            all_logits_chunks.append(gather_across_gpus(pred).cpu())
            all_attr_gt.append(gather_across_gpus(batch_attr).cpu())
            if pred_mode == "both":
                all_obj_gt.append(gather_across_gpus(batch_obj).cpu())
                all_comp_gt.append(gather_across_gpus(batch_target).cpu())

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_sum += loss.item()

            count = torch.tensor(batch_target.shape[0], device=loss.device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_correct, op=dist.ReduceOp.SUM)
            correct_count += batch_correct.item()
            total_count += count.item()

            if (idx + 1) % 10 == 0:
                print(correct_count, total_count)
                print(f"Iter {idx + 1}/{len(dataloader)}: Top1 Acc = {correct_count / total_count * 100:.2f}%")

    all_logits = torch.cat(all_logits_chunks, dim=0)
    all_attr_gt = torch.cat(all_attr_gt)
    print("eval total count : ", total_count)
    loss_avg = loss_sum / total_count

    if pred_mode == "both":
        return all_logits, all_attr_gt, torch.cat(all_obj_gt), torch.cat(all_comp_gt), loss_avg
    return all_logits, all_attr_gt, loss_avg


# --------------------------------------------------------------------------- #
# test_bias
# --------------------------------------------------------------------------- #


def test_bias(model, val_dataset, test_dataset, config):
    model.eval()
    module = _get_module(model)
    original_save_features = getattr(module, "save_features", False)
    module.save_features = True

    try:
        (open_pairs, open_idx2pair_ids, open_idx2pair_names,
         seen_mask, closed_mask) = _build_pair_indices(val_dataset)
        print('Number of seen pairs: %d' % len(seen_mask.nonzero()))
        print('Number of closed pairs: %d' % len(closed_mask.nonzero()))
        print(seen_mask.nonzero())

        # --- Val eval ---
        val_dataloader = _build_ddp_dataloader(val_dataset, config)
        all_c_logits, all_attr_gt, all_obj_gt, all_comp_gt, total_count = _collect_ow_logits(
            model, val_dataloader, config,
        )
        print("eval total count : ", total_count)

        is_seen_sample = seen_mask[all_comp_gt].bool()
        num_seen = is_seen_sample.sum().item()
        num_unseen = (~is_seen_sample).sum().item()
        print(f"Num seen {num_seen} | Num unseen {num_unseen}")

        # --- Bias grid search on val ---
        bias_candidates = np.linspace(
            getattr(config, 'bias_grid_min', 0.0),
            getattr(config, 'bias_grid_max', 0.2),
            num=getattr(config, 'bias_grid_num', 50),
        )
        best, val_seen_accs, val_unseen_accs, val_hms = _bias_grid_search(
            all_c_logits, ~seen_mask,
            [all_attr_gt, all_obj_gt, all_comp_gt],
            open_idx2pair_ids, bias_candidates, is_seen_sample,
        )

        print("\nBias Calibration Done!")
        print("-" * 71)
        print(f"Best bias: {best.bias:.4f}")
        print(f"Best Harmonic Mean (HM): {best.hm * 100:.2f}%")
        print(f"Best Seen Acc:     {best.seen_acc * 100:.2f}% (Verb: {best.seen_verb_acc * 100:.2f}%, Obj: {best.seen_obj_acc * 100:.2f}%)")
        print(f"Best Unseen Acc:   {best.unseen_acc * 100:.2f}% (Verb: {best.unseen_verb_acc * 100:.2f}%, Obj: {best.unseen_obj_acc * 100:.2f}%)")
        print("-" * 71)

        # --- Test eval ---
        print("\nTest Start!")
        test_dataloader = _build_ddp_dataloader(test_dataset, config)
        all_c_logits, all_attr_gt, all_obj_gt, all_comp_gt, total_count = _collect_ow_logits(
            model, test_dataloader, config,
        )
        print("eval total count : ", total_count)

        is_seen_sample = seen_mask[all_comp_gt].bool()
        num_seen = is_seen_sample.sum().item()
        num_unseen = (~is_seen_sample).sum().item()
        print(f"Num seen {num_seen} | Num unseen {num_unseen}")

        all_gts = [all_attr_gt, all_obj_gt, all_comp_gt]
        biased_logits = all_c_logits.clone()

        print("Unbiased Test ACC -------------------------")
        current_hm, seen_acc, unseen_acc = get_accs(biased_logits, open_idx2pair_ids, all_gts, is_seen_sample)
        seen_macro_cg, unseen_macro_cg = get_macro_cg(
            biased_logits, open_idx2pair_names, all_comp_gt,
            all_attr_gt, all_obj_gt, is_seen_sample,
            val_dataset.attr2idx, val_dataset.obj2idx,
        )
        print(f"Seen Macro CG: {seen_macro_cg:.4f}")
        print(f"Unseen Macro CG: {unseen_macro_cg:.4f}")
        print("-------------------------------------\n")

        print(f"Use Bias : {best.bias}")

        print("Biased Test ACC -------------------------")
        biased_logits[:, ~seen_mask] += best.bias
        get_accs(biased_logits, open_idx2pair_ids, all_gts, is_seen_sample)
        seen_macro_cg, unseen_macro_cg = get_macro_cg(
            biased_logits, open_idx2pair_names, all_comp_gt,
            all_attr_gt, all_obj_gt, is_seen_sample,
            val_dataset.attr2idx, val_dataset.obj2idx,
        )
        print(f"Seen Macro CG: {seen_macro_cg:.4f}")
        print(f"Unseen Macro CG: {unseen_macro_cg:.4f}")
        print("-------------------------------------\n")

        print("Biased Test ACC (CLOSED world) -------------------------")
        biased_logits[:, ~closed_mask] = 0.0
        get_accs(biased_logits, open_idx2pair_ids, all_gts, is_seen_sample)
        seen_macro_cg, unseen_macro_cg = get_macro_cg(
            biased_logits, open_idx2pair_names, all_comp_gt,
            all_attr_gt, all_obj_gt, is_seen_sample,
            val_dataset.attr2idx, val_dataset.obj2idx,
        )
        print(f"Seen Macro CG: {seen_macro_cg:.4f}")
        print(f"Unseen Macro CG: {unseen_macro_cg:.4f}")
        print("-------------------------------------\n")

        draw_seen_unseen_curve(
            bias_candidates,
            [val_hms, val_seen_accs, val_unseen_accs],
            [current_hm, seen_acc, unseen_acc],
            [best.bias, best.hm],
            config.logpath,
        )

        return biased_logits, all_attr_gt, all_obj_gt, all_comp_gt, None
    finally:
        module.save_features = original_save_features


# --------------------------------------------------------------------------- #
# test_auc
# --------------------------------------------------------------------------- #


def test_auc(model, test_dataset, config):
    model.eval()
    module = _get_module(model)
    original_save_features = getattr(module, "save_features", False)
    module.save_features = True

    try:
        (open_pairs, open_idx2pair_ids, open_idx2pair_names,
         seen_mask, closed_mask) = _build_pair_indices(test_dataset)

        # Build OW ↔ CW composition index mapping
        total_pairs_set = set(test_dataset.train_pairs + test_dataset.val_pairs + test_dataset.test_pairs)
        train_pairs_set = set(test_dataset.train_pairs)
        ow_comp_to_cw_comp, cw_seen_mask = {}, []
        for i, pair in enumerate(open_pairs):
            if pair in total_pairs_set:
                ow_comp_to_cw_comp[i] = len(cw_seen_mask)
                cw_seen_mask.append(pair in train_pairs_set)
        cw_seen_mask = torch.BoolTensor(cw_seen_mask)
        cw_comp_to_ow_comp = {v: k for k, v in ow_comp_to_cw_comp.items()}

        print('Number of seen pairs: %d' % len(seen_mask.nonzero()))
        print('Number of closed pairs: %d' % len(closed_mask.nonzero()))
        print('Number of ow_comp_to_cw_comp pairs: %d' % len(ow_comp_to_cw_comp))
        print('Number of cw_seen_mask pairs: %d' % len(cw_seen_mask.nonzero()))
        print(seen_mask.nonzero())

        # --- Test eval ---
        dataloader = _build_ddp_dataloader(test_dataset, config)
        all_c_logits, all_attr_gt, all_obj_gt, all_comp_gt, total_count = _collect_ow_logits(
            model, dataloader, config,
        )
        print("eval total count : ", total_count)

        is_seen_sample = seen_mask[all_comp_gt].bool()
        num_seen = is_seen_sample.sum().item()
        num_unseen = (~is_seen_sample).sum().item()
        print(f"Num seen {num_seen} | Num unseen {num_unseen}")

        all_gts = [all_attr_gt, all_obj_gt, all_comp_gt]

        print("Unbiased Open World Test ACC -------------------------")
        get_accs(all_c_logits.clone(), open_idx2pair_ids, all_gts, is_seen_sample)
        print("-------------------------------------\n")

        # Map OW → CW comp indices
        all_cw_comp_gt = torch.tensor(
            [ow_comp_to_cw_comp[i.item()] for i in all_comp_gt],
            dtype=torch.long, device=all_comp_gt.device,
        )
        unbiased_closed_logits = all_c_logits.clone()[:, closed_mask]

        predictions = torch.argmax(unbiased_closed_logits, dim=-1)
        unseen_matches = (predictions[~is_seen_sample] == all_cw_comp_gt[~is_seen_sample]).bool()
        seen_matches = (predictions[is_seen_sample] == all_cw_comp_gt[is_seen_sample]).bool()

        bias_candidates = np.linspace(
            getattr(config, 'auc_bias_grid_min', -0.2),
            getattr(config, 'auc_bias_grid_max', 0.2),
            num=getattr(config, 'auc_bias_grid_num', 50),
        )
        bias_candidates = np.array([-5.0] + bias_candidates.tolist() + [5.0])
        seen_match_max = float(seen_matches.float().mean())
        unseen_match_max = float(unseen_matches.float().mean())

        # Grid search — predictions are CW indices; map to (attr_id, obj_id) via OW mapping.
        cw_idx2pair = {
            i: open_idx2pair_ids[cw_comp_to_ow_comp[i]]
            for i in range(cw_seen_mask.numel())
        }
        best, biased_seen_accs, biased_unseen_accs, biased_hms = _bias_grid_search(
            unbiased_closed_logits, ~cw_seen_mask,
            [all_attr_gt, all_obj_gt, all_cw_comp_gt],
            cw_idx2pair, bias_candidates, is_seen_sample,
        )

        biased_seen_accs = list(biased_seen_accs) + [seen_match_max]
        biased_unseen_accs = list(biased_unseen_accs) + [unseen_match_max]
        seen_accuracy = np.array(biased_seen_accs)
        unseen_accuracy = np.array(biased_unseen_accs)
        area = np.trapz(seen_accuracy, unseen_accuracy)
        print(area)
        print(seen_accuracy, unseen_accuracy)

        sorted_points = sorted(zip(seen_accuracy, unseen_accuracy))
        seen_accuracy_sorted = [p[0] for p in sorted_points]
        unseen_accuracy_sorted = [p[1] for p in sorted_points]
        area = np.trapz(seen_accuracy_sorted, unseen_accuracy_sorted)
        print(area)

        from sklearn.metrics import auc
        area = auc(seen_accuracy_sorted, unseen_accuracy_sorted)

        print("\nTest Closed Bias Calibration Done!")
        print("-" * 71)
        print(f"Best bias: {best.bias:.4f}")
        print(f"AUC: {area:.4f}")
        print(f"Best Seen Acc:  {max(seen_accuracy_sorted) * 100:.2f}%)")
        print(f"Best Unseen Acc:  {max(unseen_accuracy_sorted) * 100:.2f}%)")
        print(f"Best Harmonic Mean (HM): {best.hm * 100:.2f}%")
        print(f"Best Seen Acc w/ HM:     {best.seen_acc * 100:.2f}% (Verb: {best.seen_verb_acc * 100:.2f}%, Obj: {best.seen_obj_acc * 100:.2f}%)")
        print(f"Best Unseen Acc w/ HM:   {best.unseen_acc * 100:.2f}% (Verb: {best.unseen_verb_acc * 100:.2f}%, Obj: {best.unseen_obj_acc * 100:.2f}%)")
        print("-" * 71)
    finally:
        module.save_features = original_save_features
