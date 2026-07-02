import torch.distributed as dist
import torch
from collections import Counter
import json

from engine.test_engine import predict_logits_ddp
from engine.utils import write_and_print

from itertools import product


def evaluate_only_verb(model, dataset, config, log_file=None):
    model.eval()

    if dist.is_available() and dist.is_initialized():
        all_logits, all_attr_gt, loss_avg = predict_logits_ddp(
            model, dataset, config)

    print(all_logits.shape)
    print(all_attr_gt.shape)

    top1_pred = all_logits.argmax(dim=1)
    top1_acc = (top1_pred == all_attr_gt).float().mean().item()
    print("Top-1 Acc:", top1_acc)

    K = 5
    topk_pred = all_logits.topk(K, dim=1).indices
    topk_match = (topk_pred == all_attr_gt.unsqueeze(1)).any(dim=1)
    top5_acc = topk_match.float().mean().item()
    print("Top-5 Acc:", top5_acc)

    test_stats = {'top1_acc': top1_acc, 'top5_acc': top5_acc}

    if log_file is not None:
        formatted_stats = {k: round(v, 2) if isinstance(v, float) else v for k, v in test_stats.items()}
        write_and_print(json.dumps(formatted_stats, indent=2), log_file)

    dist.barrier()
    model.train()
    return loss_avg, test_stats


def cal_accs(all_logits, total_idx2pair, seen_pair_mask, gt_labels):
    all_ow_comp_gt, all_attr_gt, all_obj_gt = gt_labels

    top1_comp_pred = all_logits.argmax(dim=1)
    top1_verb_pred, top1_obj_pred = [], []
    for c_pred in top1_comp_pred:
        v_pred, o_pred = total_idx2pair[c_pred.item()]
        top1_verb_pred.append(v_pred)
        top1_obj_pred.append(o_pred)
    top1_verb_pred = torch.Tensor(top1_verb_pred)
    top1_obj_pred = torch.Tensor(top1_obj_pred)

    seen_comp_acc = (top1_comp_pred[seen_pair_mask] == all_ow_comp_gt[seen_pair_mask]).float().mean().item()
    seen_verb_acc = (top1_verb_pred[seen_pair_mask] == all_attr_gt[seen_pair_mask]).float().mean().item()
    seen_obj_acc = (top1_obj_pred[seen_pair_mask] == all_obj_gt[seen_pair_mask]).float().mean().item()

    unseen_comp_acc = (top1_comp_pred[~seen_pair_mask] == all_ow_comp_gt[~seen_pair_mask]).float().mean().item()
    unseen_verb_acc = (top1_verb_pred[~seen_pair_mask] == all_attr_gt[~seen_pair_mask]).float().mean().item()
    unseen_obj_acc = (top1_obj_pred[~seen_pair_mask] == all_obj_gt[~seen_pair_mask]).float().mean().item()

    assert len(top1_comp_pred[seen_pair_mask]) + len(top1_comp_pred[~seen_pair_mask]) == len(all_logits)

    return seen_comp_acc, seen_verb_acc, seen_obj_acc, unseen_comp_acc, unseen_verb_acc, unseen_obj_acc


def cal_accs_with_component_logits(all_v_logits, all_o_logits, seen_pair_mask, gt_labels):
    all_ow_comp_gt, all_attr_gt, all_obj_gt = gt_labels

    top1_verb_pred = all_v_logits.argmax(dim=1)
    top1_obj_pred = all_o_logits.argmax(dim=1)

    seen_verb_acc = (top1_verb_pred[seen_pair_mask] == all_attr_gt[seen_pair_mask]).float().mean().item()
    seen_obj_acc = (top1_obj_pred[seen_pair_mask] == all_obj_gt[seen_pair_mask]).float().mean().item()

    unseen_verb_acc = (top1_verb_pred[~seen_pair_mask] == all_attr_gt[~seen_pair_mask]).float().mean().item()
    unseen_obj_acc = (top1_obj_pred[~seen_pair_mask] == all_obj_gt[~seen_pair_mask]).float().mean().item()

    return seen_verb_acc, seen_obj_acc, unseen_verb_acc, unseen_obj_acc


def evaluate_composition(model, dataset, config, pairs, p_v_o=None, log_file=None, idx2label=None, collate_fn=None):
    if idx2label is not None:
        idx2attr, idx2obj = idx2label
    model.eval()

    if dist.is_available() and dist.is_initialized():
        if fake_evidence_validation:
            all_logits, all_attr_gt, all_obj_gt, all_comp_gt, loss_avg, all_fake_logits = predict_logits_ddp(model, dataset, config, collate_fn=collate_fn)
            all_fake_cond_verb_logits, all_fake_cond_obj_logits, all_true_cond_verb_logits, all_true_cond_obj_logits = all_fake_logits
        else:
            all_logits, all_attr_gt, all_obj_gt, all_comp_gt, loss_avg = predict_logits_ddp(model, dataset, config, collate_fn=collate_fn)

    print(all_logits.shape)
    print(all_comp_gt.shape)

    train_pairs, total_pairs = pairs
    total_idx2pair = {i: elem for i, elem in enumerate(total_pairs)}
    total_pair2idx = {elem: i for i, elem in enumerate(total_pairs)}

    seen_pair_mask = []
    all_ow_comp_gt = []
    for attr_label, obj_label in zip(all_attr_gt, all_obj_gt):
        pair = (attr_label.item(), obj_label.item())
        if pair in set(train_pairs):
            seen_pair_mask.append(True)
        else:
            seen_pair_mask.append(False)
        all_ow_comp_gt.append(total_pair2idx[pair])
    seen_pair_mask = torch.BoolTensor(seen_pair_mask)
    all_ow_comp_gt = torch.Tensor(all_ow_comp_gt)

    gt_labels = [all_ow_comp_gt, all_attr_gt, all_obj_gt]
    seen_comp_acc, seen_verb_acc, seen_obj_acc, unseen_comp_acc, unseen_verb_acc, unseen_obj_acc = cal_accs(all_logits, total_idx2pair, seen_pair_mask, gt_labels)

    if getattr(config, "val_with_confusion", False):
        # FSP and FCP analysis
        if p_v_o is not None:
            p_v_o_on_o, p_v_o_on_v, p_v_o_on_v_o = p_v_o
            all_pair_idxes = list(product(range(len(idx2attr)), range(len(idx2obj))))
            seen_mask_ow = torch.BoolTensor([True if elem in train_pairs else False for elem in all_pair_idxes])
            seen_p_v_o_on_v_o = p_v_o_on_v_o.view(-1)[seen_mask_ow]
            thres_vo = seen_p_v_o_on_v_o.mean() + seen_p_v_o_on_v_o.std()

            FCP, FCP_verb_collapse, FCP_object_collapse, FCP_dual_collapse = 0, 0, 0, 0
            FSP = 0
            total_unseen = 0
            unseen_cor = 0
            unseen_wrong = 0

            for logit, v_gt, o_gt in zip(all_logits, all_attr_gt, all_obj_gt):
                is_seen = (v_gt.item(), o_gt.item()) in train_pairs
                if not is_seen:
                    c_pred = logit.argmax(dim=-1)
                    v_pred, o_pred = all_pair_idxes[c_pred.item()]

                    if all_pair_idxes[c_pred] == (v_gt.item(), o_gt.item()):
                        unseen_cor += 1
                    else:
                        is_pred_unseen_as_seen = (v_pred, o_pred) in train_pairs
                        if is_pred_unseen_as_seen:
                            v_pred, o_pred = all_pair_idxes[c_pred.item()]
                            if p_v_o_on_v_o[v_pred, o_pred] > thres_vo:
                                FCP += 1
       
                                if o_pred == o_gt.item() and v_pred != v_gt.item():
                                    FCP_verb_collapse += 1
                                if v_pred == v_gt.item() and o_pred != o_gt.item():
                                    FCP_object_collapse += 1
                                if o_pred != o_gt.item() and v_pred != v_gt.item():
                                    FCP_dual_collapse += 1

                            FSP += 1

                        unseen_wrong += 1

                    total_unseen += 1

            print(f"Unseen accuracy               : {unseen_cor/total_unseen*100:.2f}%")
            print(f"Unseen error rate             : {unseen_wrong/total_unseen*100:.2f}%")
            print(f"N total_unseen                : {total_unseen}")
            print(f"N unseen_wrong                : {unseen_wrong}")

            print(f"FSP (False Seen Prediction)   : {FSP/unseen_wrong*100:.2f}%   (N={FSP})")
            print(f"FCP (False Co-occurrence Prediction, th={thres_vo:.2f}): {FCP/unseen_wrong*100:.2f}%   (N={FCP})")
            print(f"  Verb-collapse   / FCP : {FCP_verb_collapse/FCP*100:.2f}%")
            print(f"  Object-collapse / FCP : {FCP_object_collapse/FCP*100:.2f}%")
            print(f"  Dual-collapse   / FCP : {FCP_dual_collapse/FCP*100:.2f}%")

    print("[SEEN (OW)]")
    print(f"Top-1 Comp Acc: {seen_comp_acc:.4f}")
    print(f"Top-1 Verb Acc: {seen_verb_acc:.4f}")
    print(f"Top-1 Obj Acc: {seen_obj_acc:.4f}")
    print("\n[UNSEEN (OW)]")
    print(f"Top-1 Comp Acc: {unseen_comp_acc:.4f}")
    print(f"Top-1 Verb Acc: {unseen_verb_acc:.4f}")
    print(f"Top-1 Obj Acc: {unseen_obj_acc:.4f}")


    test_stats = {'seen_comp_acc': seen_comp_acc, 'seen_verb_acc': seen_verb_acc,
                  "seen_obj_acc": seen_obj_acc, "unseen_comp_acc": unseen_comp_acc,
                  "unseen_verb_acc": unseen_verb_acc, "unseen_obj_acc": unseen_obj_acc,
                  "hm_comp_acc": 2 * seen_comp_acc * unseen_comp_acc / (seen_comp_acc + unseen_comp_acc + 1.0e-6),
                  "hm_verb_acc": 2 * seen_verb_acc * unseen_verb_acc / (seen_verb_acc + unseen_verb_acc + 1.0e-6),
                  "hm_obj_acc": 2 * seen_obj_acc * unseen_obj_acc / (seen_obj_acc + unseen_obj_acc + 1.0e-6)}

    if log_file is not None:
        formatted_stats = {k: round(v, 2) if isinstance(v, float) else v for k, v in test_stats.items()}
        write_and_print(json.dumps(formatted_stats, indent=2), log_file)

    dist.barrier()
    model.train()
    return loss_avg, test_stats


def evaluate(model, dataset, config, pairs=None, p_v_o=None, log_file=None, idx2label=None, collate_fn=None):
    if hasattr(dataset, "train_pairs"):
        return evaluate_composition(model, dataset, config, pairs, p_v_o=p_v_o, log_file=log_file, idx2label=idx2label, collate_fn=collate_fn)
    else:
        return evaluate_only_verb(model, dataset, config, log_file=log_file, collate_fn=collate_fn)
