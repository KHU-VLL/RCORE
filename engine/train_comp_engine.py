import os
import time as lib_time
from collections import defaultdict
from itertools import product

import numpy as np
import torch
import tqdm

from utils.loss import *
from engine.utils import *
from engine.evaluate_helper import evaluate


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
# Main training loop (pure C2C baseline)
# --------------------------------------------------------------------------- #


def train(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
          scaler, imgnet_test_dataset=None, collate_fn=None):
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

    p_v_o_on_v, p_v_o_on_o, p_v_o_on_v_o = cal_conditional(attr2idx, obj2idx, 'train', train_dataset)

    autocast_dtype = (torch.bfloat16 if getattr(config, "use_flash_attn", False)
                     else torch.float16)

    with open(os.path.join(config.save_path, 'log.txt'), 'w') as log_training:
        for i in range(config.epoch_start, config.epochs):
            progress_bar = tqdm.tqdm(total=len(train_dataloader), desc=f"epoch {i + 1:3d}")
            start_time = lib_time.time()
            sampler.set_epoch(i)

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

                with torch.amp.autocast(device_type='cuda', enabled=True, dtype=autocast_dtype):
                    if getattr(config, "use_compcos", False):
                        com_logits = model(batch_img)
                        loss_com = loss_fn(com_logits * config.cosine_scale, batch_target)
                        loss_verb = loss_obj = torch.tensor(0.0, device=batch_target.device)
                        loss = loss_com
                    else:
                        verb_logits, obj_logits, com_logits = model(batch_img)

                        loss_verb = loss_fn(verb_logits * config.cosine_scale, batch_verb)
                        loss_obj = loss_fn(obj_logits * config.cosine_scale, batch_obj)

                        closed_com_logits = com_logits[:, seen_mask]
                        loss_com = loss_fn(closed_com_logits * config.cosine_scale, batch_target)

                        loss = 0.2 * (loss_verb + loss_obj) + loss_com

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
