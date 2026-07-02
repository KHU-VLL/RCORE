import os
import time as lib_time
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import tqdm

from utils.loss import *
from engine.utils import *
from engine.evaluate_helper import evaluate
from utils.fame import FAME
from utils.cpr import apply_CPR


# --------------------------------------------------------------------------- #
# Config / schedules
# --------------------------------------------------------------------------- #


@dataclass
class LossScaleSchedule:
    """Linear warm-up from 0 → `weight` between epochs `[start, end)`."""

    weight: float
    start: int
    end: int

    def __call__(self, epoch: int) -> float:
        if epoch < self.start:
            return 0.0
        if epoch < self.end:
            return (epoch - self.start) / (self.end - self.start) * self.weight
        return self.weight


@dataclass
class TrainingHyperparams:
    # CPR
    CPR_aug_prob: float
    CPR_alpha: float
    CPR_lambda: float
    CPR_OW_com_loss_scale: float
    CPR_OW_com_loss_start: int
    CW_com_loss_scale: float

    # Temporal shuffle 
    temporal_modeling_shuffle: bool
    shuffle_loss_schedule: LossScaleSchedule

    # Co-occurrence penalty
    cooc_mask_type: str
    cooc_penalty_margin: float
    cooc_penalty_schedule: LossScaleSchedule
    train_cooc_thres_std: float

    # FAME augmentation
    use_FAME: bool
    fame_beta: float

    # AMP dtype
    use_flash_attn: bool

    @classmethod
    def from_config(cls, config):
        penalty_start = getattr(config, 'penalty_loss_scale_start', 5)
        penalty_end = getattr(config, 'penalty_loss_scale_end', 10)
        return cls(
            CPR_aug_prob=getattr(config, 'CPR_aug_prob', 0.5),
            CPR_alpha=getattr(config, 'CPR_alpha', 1.0),
            CPR_lambda=getattr(config, 'CPR_lambda', 1.0),
            CPR_OW_com_loss_scale=getattr(config, 'CPR_OW_com_loss_scale', 0.1),
            CPR_OW_com_loss_start=getattr(config, 'CPR_OW_com_loss_start', -1),
            CW_com_loss_scale=getattr(config, 'CW_com_loss_scale', 1.0),
            temporal_modeling_shuffle=getattr(config, 'temporal_modeling_shuffle', False),
            shuffle_loss_schedule=LossScaleSchedule(
                weight=getattr(config, 'temporal_modeling_shuffle_loss_scale', 0.5),
                start=penalty_start,
                end=penalty_end,
            ),
            cooc_mask_type=getattr(config, 'cooc_mask_type', "p_v_o_on_o"),
            cooc_penalty_margin=getattr(config, 'cooc_penalty_margin', 2.0),
            cooc_penalty_schedule=LossScaleSchedule(
                weight=getattr(config, 'cooc_penalty_loss_scale', 0.2),
                start=getattr(config, 'cooc_penalty_loss_scale_start', 5),
                end=getattr(config, 'cooc_penalty_loss_scale_end', 10),
            ),
            train_cooc_thres_std=getattr(config, "train_cooc_threshold", 1.0),
            use_FAME=getattr(config, 'use_FAME', False),
            fame_beta=getattr(config, 'fame_beta', 0.4),
            use_flash_attn=getattr(config, "use_flash_attn", False),
        )


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #


def get_state_dict(model):
    return model.state_dict()


def _build_dataloader(train_dataset, config, collate_fn):
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader, sampler


def _build_seen_mask(train_pairs_set, open_world_pairs, open_world):
    if open_world:
        return torch.BoolTensor([True] * len(open_world_pairs))
    return torch.BoolTensor([pair in train_pairs_set for pair in open_world_pairs])


def _build_cooc_lookups(open_world_pairs, train_pairs_set,
                        p_v_o_on_o, p_v_o_on_v, p_v_o_on_v_o):
    n = len(open_world_pairs)
    cooc_o = torch.zeros(n)
    cooc_v = torch.zeros(n)
    cooc_vo = torch.zeros(n)
    for idx, (attr, obj) in enumerate(open_world_pairs):
        if (attr, obj) in train_pairs_set:
            cooc_o[idx] = p_v_o_on_o[attr, obj]
            cooc_v[idx] = p_v_o_on_v[attr, obj]
            cooc_vo[idx] = p_v_o_on_v_o[attr, obj]
    return cooc_o, cooc_v, cooc_vo


def _compute_thresholds(cooc_o, cooc_v, cooc_vo, std_mult):
    def _thr(x):
        pos = x[x > 0]
        return pos.mean() + pos.std() * std_mult
    return _thr(cooc_v), _thr(cooc_o), _thr(cooc_vo)


def _build_fame_operator(params):
    if not params.use_FAME:
        return None
    return FAME(
        crop_size=224,
        beta=params.fame_beta,
        device='cuda',
    )


def _select_high_cooc_mask(cooc_mask_type, cooc_o, cooc_v, cooc_vo,
                           thres_o, thres_v, thres_vo):
    if cooc_mask_type == "p_v_o_on_o_and_v":
        return (cooc_o > thres_o) & (cooc_v > thres_v)
    if cooc_mask_type == "p_v_o_on_total":
        return cooc_vo > thres_vo
    raise NotImplementedError(f"Unknown cooc_mask_type: {cooc_mask_type}")


# --------------------------------------------------------------------------- #
# Loss helpers
# --------------------------------------------------------------------------- #


def _compute_cooc_penalty(new_com_logits, batch_verb, batch_obj_idx,
                          comp_verb_target, comp_obj_target,
                          batch_OW_target, batch_new_target,
                          base_high_cooc_mask, new_seen_mask,
                          cooc_penalty_margin, cosine_scale):
    log_probs = F.log_softmax(new_com_logits * cosine_scale, dim=-1)
    device = log_probs.device

    high_cooc_mask = base_high_cooc_mask[new_seen_mask].unsqueeze(0).expand_as(log_probs).to(device)

    shared_v_mask = (batch_verb.unsqueeze(1) == comp_verb_target.unsqueeze(0))
    shared_o_mask = (batch_obj_idx.unsqueeze(1) == comp_obj_target.unsqueeze(0))
    shared_vo_mask = (shared_v_mask | shared_o_mask).to(device)

    top2_indices = torch.topk(batch_OW_target, k=2, dim=1).indices
    non_gt_mask = torch.ones_like(log_probs, dtype=torch.bool)
    non_gt_mask.scatter_(1, top2_indices, False)

    penalty_mask = high_cooc_mask & shared_vo_mask & non_gt_mask

    gt_logits = torch.gather(log_probs, 1, batch_new_target.unsqueeze(1))
    penalties = F.relu(log_probs - gt_logits + cooc_penalty_margin)
    return (penalties * penalty_mask.float()).sum(dim=1).mean()


def _compute_shuffle_loss(shuffled_verb_outputs, cosine_scale):
    """Return (shuffle_loss, cos_with_random_feat_for_logging)."""
    shuffled_verb_cosine_loss, shuffled_verb_logits, cos_with_random_feat = shuffled_verb_outputs
    log_probs = F.log_softmax(shuffled_verb_logits * cosine_scale, dim=-1)
    probs = torch.exp(log_probs)
    entropy_loss = -torch.sum(probs * log_probs, dim=-1).mean()
    return shuffled_verb_cosine_loss.mean() - entropy_loss, cos_with_random_feat


# --------------------------------------------------------------------------- #
# Checkpoint / eval helpers
# --------------------------------------------------------------------------- #


def _save_ckpt(model, optimizer, lr_scheduler, scaler, save_path, epoch, best=False):
    save_checkpoint({
        'state_dict': get_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
    }, save_path, epoch, best=best)


def _run_val_and_maybe_save_best(model, val_dataset, config, train_pairs, open_world_pairs,
                                 p_v_o_on_o, p_v_o_on_v, p_v_o_on_v_o,
                                 idx2attr, idx2obj, collate_fn, log_training,
                                 epoch, optimizer, lr_scheduler, scaler, best_state):
    write_and_print("Evaluating val dataset:", log_training)
    loss_avg, val_result = evaluate(
        model, val_dataset, config, [train_pairs, open_world_pairs],
        p_v_o=[p_v_o_on_o, p_v_o_on_v, p_v_o_on_v_o],
        log_file=log_training,
        idx2label=[idx2attr, idx2obj],
        collate_fn=collate_fn,
    )
    result = " | ".join(f"{k}  {round(v, 4)}" for k, v in val_result.items())
    write_and_print(result, log_training)
    write_and_print(f"Loss average on val dataset: {loss_avg}", log_training)
    print(config.save_path)

    if config.best_model_metric == "best_loss":
        if isinstance(loss_avg, torch.Tensor):
            loss_avg = loss_avg.cpu().float()
        is_best = loss_avg < best_state['best_loss']
        if is_best:
            best_state['best_loss'] = loss_avg
    else:
        is_best = val_result[config.best_model_metric] > best_state['best_metric']
        if is_best:
            best_state['best_metric'] = val_result[config.best_model_metric]

    if is_best:
        write_and_print("find best!", log_training)
        _save_ckpt(model, optimizer, lr_scheduler, scaler, config.save_path, epoch, best=True)


def _resolve_final_ckpt(save_path):
    """Prefer best.pt, then epoch_resume.pt; fall back to `<save_path>/0/…`."""
    for name in ("best.pt", "epoch_resume.pt"):
        primary = os.path.join(save_path, name)
        if os.path.exists(primary):
            return primary
        fallback = os.path.join(save_path, "0", name)
        if os.path.exists(fallback):
            return fallback
    return os.path.join(save_path, "epoch_resume.pt")  # last-ditch (will raise on load)


def _run_final_test(model, test_dataset, config, train_pairs, open_world_pairs,
                    collate_fn, log_training):
    torch.cuda.empty_cache()
    write_and_print("Evaluating test dataset on Open World", log_training)
    ckpt_path = _resolve_final_ckpt(config.save_path)
    ckpt = torch.load(ckpt_path)['state_dict']
    model.module.load_state_dict(ckpt, strict=False)

    loss_avg, val_result = evaluate(
        model, test_dataset, config, [train_pairs, open_world_pairs],
        log_file=log_training, collate_fn=collate_fn,
    )
    result = " | ".join(f"{k}  {round(v, 4)}" for k, v in val_result.items())
    write_and_print(result, log_training)
    write_and_print(f"Final Loss average on test dataset: {loss_avg}", log_training)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #


def train(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
          scaler, collate_fn=None):
    params = TrainingHyperparams.from_config(config)
    train_dataloader, sampler = _build_dataloader(train_dataset, config, collate_fn)

    model.train()
    best_state = {'best_loss': float('inf'), 'best_metric': float('-inf')}
    loss_fn = CrossEntropyLoss()

    # Vocabulary and pair indices
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    idx2attr = {v: k for k, v in attr2idx.items()}
    idx2obj = {v: k for k, v in obj2idx.items()}
    train_pairs = [(attr2idx[a], obj2idx[o]) for a, o in train_dataset.train_pairs]
    open_world_pairs = list(product(attr2idx.values(), obj2idx.values()))
    train_pairs_set = set(train_pairs)
    open_world_pair2idx = {p: i for i, p in enumerate(open_world_pairs)}

    seen_mask = _build_seen_mask(train_pairs_set, open_world_pairs, config.open_world)
    print("SEEN Mask nonzero sum : ", seen_mask.sum().item())

    if getattr(config, "use_compcos", False):
        model.module.init_compcos_metadata(train_pairs, open_world_pairs, open_world_pair2idx)

    print(f"CPR - CW loss scale : {params.CW_com_loss_scale}, "
          f"OW loss for new actions scale : {params.CPR_OW_com_loss_scale}")
    print(f"cooc_penalty_loss_scale: {params.cooc_penalty_schedule.weight}")
    print(f"cooc_mask_type: {params.cooc_mask_type}")

    p_v_o_on_v, p_v_o_on_o, p_v_o_on_v_o = cal_conditional(attr2idx, obj2idx, 'train', train_dataset)
    cooc_o, cooc_v, cooc_vo = _build_cooc_lookups(
        open_world_pairs, train_pairs_set, p_v_o_on_o, p_v_o_on_v, p_v_o_on_v_o,
    )
    thres_v, thres_o, thres_vo = _compute_thresholds(
        cooc_o, cooc_v, cooc_vo, params.train_cooc_thres_std,
    )
    print(f"thres_v : {thres_v:.2f}, thres_o : {thres_o:.2f}, thres_vo : {thres_vo:.2f}")

    # `high_cooc_mask` base doesn't depend on batch — compute once.
    base_high_cooc_mask = _select_high_cooc_mask(
        params.cooc_mask_type, cooc_o, cooc_v, cooc_vo, thres_o, thres_v, thres_vo,
    )

    fame_operator = _build_fame_operator(params)
    print("CPR operator : ", apply_CPR.__name__)

    autocast_dtype = torch.bfloat16 if params.use_flash_attn else torch.float16

    with open(os.path.join(config.save_path, 'log.txt'), 'w') as log_training:
        for i in range(config.epoch_start, config.epochs):
            progress_bar = tqdm.tqdm(total=len(train_dataloader), desc=f"epoch {i + 1:3d}")
            start_time = lib_time.time()
            sampler.set_epoch(i)

            shuffle_scale = params.shuffle_loss_schedule(i)
            cooc_scale = params.cooc_penalty_schedule(i)
            if params.temporal_modeling_shuffle:
                print(f"Epoch {i} : temporal_modeling_shuffle_loss_scale: {shuffle_scale}")
            print(f"Epoch {i} : cooc_penalty_loss_scale: {cooc_scale}")

            epoch_losses: dict = defaultdict(list)

            write_and_print(f"Current_lr: {optimizer.param_groups[-1]['lr']}", log_training)

            if getattr(config, "use_compcos", False):
                model.module.update_compcos_margin(i)

            for bid, batch in enumerate(train_dataloader):
                batch_img = batch[0]
                if isinstance(batch_img, torch.Tensor):
                    batch_img = batch_img.cuda()
                elif isinstance(batch_img, dict):
                    batch_img = {k: v.cuda() for k, v in batch_img.items()}

                batch_verb = batch[1].cuda()
                batch_obj = batch[2].cuda()
                batch_target = batch[3].cuda()

                # CPR — required (co-occurrence penalty depends on the outputs).
                batch_obj = F.one_hot(batch_obj, num_classes=len(obj2idx)).float()
                batch_img, new_labels = apply_CPR(
                    batch_img,
                    params.CPR_aug_prob, params.CPR_alpha, params.CPR_lambda,
                    [batch_verb, batch_obj, batch_target],
                    open_world_pair2idx,
                    fame_operator=fame_operator,
                    p_v_o=[p_v_o_on_v, p_v_o_on_o],
                    seen_mask=seen_mask,
                    return_comp_label=True,
                )
                (batch_verb, batch_obj, batch_target, batch_OW_target,
                 new_seen_mask, comp_verb_target, comp_obj_target) = new_labels
                batch_OW_target, batch_new_target = batch_OW_target

                with torch.amp.autocast(device_type='cuda', enabled=True, dtype=autocast_dtype):
                    if params.temporal_modeling_shuffle:
                        verb_logits, obj_logits, com_logits, shuffled_verb_outputs = model(batch_img)
                    else:
                        verb_logits, obj_logits, com_logits = model(batch_img)

                    loss_verb = loss_fn(verb_logits * config.cosine_scale, batch_verb)
                    loss_obj = loss_fn(obj_logits * config.cosine_scale, batch_obj)

                    new_com_logits = com_logits[:, new_seen_mask]
                    loss_OW_com = loss_fn(new_com_logits * config.cosine_scale, batch_OW_target)

                    closed_com_logits = com_logits[:, seen_mask]
                    loss_com = loss_fn(closed_com_logits * config.cosine_scale, batch_target)

                    batch_obj_idx = torch.argmax(batch_obj, dim=1)
                    loss_cooc_penalty = _compute_cooc_penalty(
                        new_com_logits, batch_verb, batch_obj_idx,
                        comp_verb_target, comp_obj_target,
                        batch_OW_target, batch_new_target,
                        base_high_cooc_mask, new_seen_mask,
                        params.cooc_penalty_margin, config.cosine_scale,
                    )

                    ow_com_scale = params.CPR_OW_com_loss_scale if i > params.CPR_OW_com_loss_start else 0.0
                    loss = (0.2 * (loss_verb + loss_obj)
                            + params.CW_com_loss_scale * loss_com
                            + ow_com_scale * loss_OW_com)

                    if params.temporal_modeling_shuffle:
                        shuffled_verb_loss, cos_with_random_feat = _compute_shuffle_loss(
                            shuffled_verb_outputs, config.cosine_scale,
                        )
                        loss += shuffle_scale * shuffled_verb_loss

                    loss += cooc_scale * loss_cooc_penalty
                    loss /= config.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_step_end = ((bid + 1) % config.gradient_accumulation_steps == 0
                               or bid + 1 == len(train_dataloader))
                if is_step_end:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                epoch_losses['train'].append(loss.item())
                epoch_losses['vv'].append(loss_verb.item())
                epoch_losses['oo'].append(loss_obj.item())
                epoch_losses['com'].append(loss_com.item())
                epoch_losses['CPR_OW_com'].append(loss_OW_com.item())
                epoch_losses['cooc_penalty'].append(loss_cooc_penalty.item())
                if params.temporal_modeling_shuffle:
                    epoch_losses['shuffled_verb_total'].append(shuffled_verb_loss.item())
                    epoch_losses['shuffled_verb_cosine'].append(cos_with_random_feat.mean().item())

                progress_bar.set_postfix({"train loss": np.mean(epoch_losses['train'][-50:])})
                progress_bar.update()

            lr_scheduler.step()
            progress_bar.close()

            elapsed_str = lib_time.strftime("%H:%M:%S", lib_time.gmtime(lib_time.time() - start_time))
            write_and_print(f"Epoch {i + 1} training time: {elapsed_str}", log_training)
            for name, vals in epoch_losses.items():
                write_and_print(f"epoch {i + 1} {name} loss {np.mean(vals)}", log_training)

            if (i + 1) % config.save_every_n == 0:
                _save_ckpt(model, optimizer, lr_scheduler, scaler, config.save_path, i)

            if i % config.eval_every_n == 0 or i + 1 == config.epochs:
                _run_val_and_maybe_save_best(
                    model, val_dataset, config, train_pairs, open_world_pairs,
                    p_v_o_on_o, p_v_o_on_v, p_v_o_on_v_o,
                    idx2attr, idx2obj, collate_fn, log_training,
                    i, optimizer, lr_scheduler, scaler, best_state,
                )

            if i + 1 == config.epochs:
                _run_final_test(
                    model, test_dataset, config, train_pairs, open_world_pairs,
                    collate_fn, log_training,
                )

    return None
