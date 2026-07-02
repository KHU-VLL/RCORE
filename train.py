import random
import os, json
import pprint
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
import torch.multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.opts import parser
from utils.loss import *
from utils import CosineAnnealingLR

from engine import train_comp_engine, train_comp_engine_penalty
from dataset.com_video_dataset import CompositionVideoDataset

from peft import LoraConfig, get_peft_model


torch.multiprocessing.set_sharing_strategy('file_system')


def setup_for_distributed(is_master, log_file_path=None):
    """Suppress non-master prints and tee master output to a log file."""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    builtin_pprint = pprint.pprint

    def _tee_to_file(writer):
        if log_file_path is None:
            return
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                writer(f)
        except Exception as e:
            builtin_print(f"Warning: Could not write to log file {log_file_path}: {e}")

    def new_print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
            _tee_to_file(lambda f: builtin_print(*args, **kwargs, file=f))

    def new_pprint(*args, **kwargs):
        if is_master:
            builtin_pprint(*args, **kwargs)
            _tee_to_file(lambda f: builtin_pprint(*args, **kwargs, stream=f))

    __builtin__.print = new_print
    # Patch the module attribute so `pprint.pprint(...)` at call sites picks it up.
    pprint.pprint = new_pprint


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for _, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def setup():
    dist.init_process_group(backend='nccl')


def cleanup():
    dist.destroy_process_group()


def resolve_save_path(local_rank, base):
    """Pick the first non-existing `<base>/<i>` directory. Rank 0 searches; result is broadcast."""
    base = base + '/'
    if local_rank == 0:
        i = 0
        candidate = base + str(i)
        while os.path.exists(candidate):
            print(f'file {candidate} already exists')
            i += 1
            candidate = base + str(i)
        save_path = candidate
        print(f'file {save_path}')
    else:
        save_path = None

    container = [save_path]
    dist.broadcast_object_list(container, src=0)
    return container[0]


def build_datasets(config):
    common = dict(
        split='compositional-split-natural',
        tdn_input='tdn' in config.arch,
        frames_duration=config.num_frames,
        open_world=config.open_world,
        split_root=getattr(config, "split_root", None),
        config=config,
    )
    train_dataset = CompositionVideoDataset(
        config.dataset_path, phase='train',
        aux_input=config.aux_input, ade_input=config.ade_input,
        **common,
    )
    val_dataset = CompositionVideoDataset(config.dataset_path, phase='val', **common)
    test_dataset = CompositionVideoDataset(config.dataset_path, phase='test', **common)
    return train_dataset, val_dataset, test_dataset


def build_model(config):
    if "internvideo_c2c_1b" in config.method:
        from model.custom_internvideo_1b import load
        return load(config.backbone, config)
    if "internvideo" in config.method:
        from model.custom_internvideo import load
        return load(config.backbone, config)
    if "c2c" in config.method:
        from model.custom_clip_c2c_prompts import load
        return load(config.backbone, config)
    raise ValueError(f"Unknown method: {config.method}")


def load_pretrain(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print("*** LOAD PRETRAIN ***")
    print("path : ", ckpt_path)
    print(msg)


def build_lora(model, config):
    """Compute LoRA target modules and wrap `model` with PEFT."""
    total_visual_layers = model.vision_encoder.get_num_layers()
    if config.method == "internvideo_c2c":
        total_llm_layers = len(model.text_encoder.transformer)
    else:  # internvideo_c2c_1b
        total_llm_layers = len(model.text_encoder.bert.encoder.layer)

    visual_start_layer = config.adapt_star_layer
    llm_start_layer = (
        config.adapt_star_layer + (total_llm_layers - total_visual_layers)
        if config.adapt_star_layer > 0 else config.adapt_star_layer
    )

    print("total_visual_layers: ", total_visual_layers)
    print("total_llm_layers: ", total_llm_layers)
    print("visual_start_layer: ", visual_start_layer)
    print("llm_start_layer: ", llm_start_layer)

    if hasattr(model, 'config'):
        model.config = vars(model.config)

    targets = []
    for i in range(visual_start_layer, total_visual_layers):
        targets.append(f"vision_encoder.blocks.{i}.attn.qkv")
        targets.append(f"vision_encoder.blocks.{i}.mlp.fc1")
        targets.append(f"vision_encoder.blocks.{i}.mlp.fc2")

    if getattr(config, "llm_lora_tuning", False):
        # `llm_start_layer` is forced to 0 when LLM LoRA tuning is enabled.
        for i in range(0, total_llm_layers):
            if config.method == "internvideo_c2c":
                targets.extend([
                    f"text_encoder.transformer.{i}.pre_norm_mha.1.qkv_proj",
                    f"text_encoder.transformer.{i}.pre_norm_mha.1.out_proj",
                    f"text_encoder.transformer.{i}.pre_norm_ffn.1",
                    f"text_encoder.transformer.{i}.pre_norm_ffn.4",
                ])
            else:  # internvideo_c2c_1b
                targets.extend([
                    f"text_encoder.bert.encoder.layer.{i}.attention.self.query",
                    f"text_encoder.bert.encoder.layer.{i}.attention.self.key",
                    f"text_encoder.bert.encoder.layer.{i}.attention.self.value",
                ])

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        modules_to_save=["c2c"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _apply_freeze_internvideo_c2c(model, config):
    """Additively unfreeze projection / c2c / coop prompt params after LoRA wrapping."""
    use_coop_prompt = getattr(config, "prompt_learn_method", None) == 'coop'
    for name, param in model.named_parameters():
        if any(k in name for k in ('projector', 'vision_align', 'c2c', 'vision_proj', 'text_proj')):
            param.requires_grad = True
        if use_coop_prompt and 'prompt_learner' in name:
            param.requires_grad = True


def _apply_freeze_clip_aim(model, config):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'video_encoder' in name:
            if any(k in name for k in ('temporal_embedding', 'ln_post', 'Adapter')):
                param.requires_grad = True
        elif 'prompt_learner' in name:
            if config.prompt_learn_method != 'coop':
                raise NotImplementedError()
            if 'ctx' in name or 'positional_embedding' in name:
                param.requires_grad = True


def _apply_freeze_default(model, config):
    prompt_learn = getattr(config, "prompt_learn_method", None)
    total_num_layers = len(model.text_encoder.resblocks) if hasattr(model, 'text_encoder') else 0

    for name, param in model.named_parameters():
        param.requires_grad = False

        if prompt_learn is not None and 'prompt_learner' in name:
            if prompt_learn != 'coop':
                raise NotImplementedError
            if 'ctx' in name or 'positional_embedding' in name:
                param.requires_grad = True
        elif 'video_encoder' in name or 'vision_encoder' in name:
            if any(k in name for k in ('temporal_embedding', 'ln_post', 'Adapter')):
                param.requires_grad = True
        elif 'c2c' in name or 'projector' in name:
            param.requires_grad = True
        elif 'text_encoder' in name:
            assert hasattr(config, "num_text_unfreeze_layer")
            if 'resblocks' in name:
                parts = name.split('.')
                layer_num = next(
                    (int(parts[i + 1]) for i, p in enumerate(parts[:-1]) if p == 'resblocks'),
                    None,
                )
                if layer_num and layer_num >= (total_num_layers - config.num_text_unfreeze_layer):
                    param.requires_grad = True


def _build_default_optimizer(config, model):
    vision_no_wd, vision_with_wd = [], []
    text_with_wd, text_no_wd = [], []
    c2c_with_wd, prompt_param = [], []

    if getattr(config, "prompt_learn_method", None) == "coop":
        for _, param in model.prompt_learner_verb.named_parameters():
            prompt_param.append(param)
        if config.pred_mode != 'single':
            for name, param in model.prompt_learner_obj.named_parameters():
                if 'token_embedding' not in name:
                    prompt_param.append(param)

    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name:
                vision_no_wd.append(param)
            elif 'Adapter' in name or 'clip_proj' in name:
                vision_with_wd.append(param)
        if 'text_encoder' in name:
            if 'resblocks' in name:
                text_with_wd.append(param)
            if 'ln_final' in name or 'text_projection' in name:
                text_no_wd.append(param)
        if 'projector_' in name:
            vision_with_wd.append(param)
        if 'c2c' in name:
            c2c_with_wd.append(param)

    assert config.text_encoding_manner == 'component', config.text_encoding_manner
    return torch.optim.AdamW(
        [
            {'params': vision_with_wd, 'lr': config.visual_lr, 'weight_decay': config.visual_wd},
            {'params': vision_no_wd,   'lr': config.visual_lr, 'weight_decay': 0.0},
            {'params': text_with_wd,   'lr': config.text_lr,   'weight_decay': config.text_wd},
            {'params': text_no_wd,     'lr': config.text_lr,   'weight_decay': 0.0},
            {'params': c2c_with_wd,    'lr': config.visual_lr, 'weight_decay': config.visual_wd},
            {'params': prompt_param,   'lr': config.text_lr,   'weight_decay': config.text_wd},
        ],
        betas=(0.9, 0.999), lr=config.visual_lr, eps=1e-8,
        weight_decay=config.visual_wd,
    )


def _build_grouped_parameters(model, weight_decay, config, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if "internvideo_c2c_1b" in config.method:
            if "vision" in name:
                group_name = "vision_" + group_name
                lr = config.visual_lr
            elif "text" in name:
                group_name = "text_" + group_name
                lr = config.text_lr
            elif "c2c" in name or 'prompt_learner' in name:
                group_name = "c2c_" + group_name
                lr = config.c2c_lr
            else:
                lr = config.visual_lr
        else:
            lr = config.visual_lr

        if group_name not in parameter_group_names:
            template = {"weight_decay": this_weight_decay, "params": [], "lr_scale": 1., "lr": lr}
            parameter_group_names[group_name] = {**template}
            parameter_group_vars[group_name] = {**template}
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def build_optimizer_and_scheduler(model, config):
    if config.method in ("internvideo_aim", "internvideo_c2c", "internvideo_c2c_1b"):
        parameters = _build_grouped_parameters(model, config.visual_wd, config=config)
        optimizer = torch.optim.AdamW(parameters, lr=config.visual_lr, weight_decay=config.visual_wd)
    else:
        optimizer = _build_default_optimizer(config, model)

    lr_scheduler = CosineAnnealingLR.WarmupCosineLR(
        optimizer=optimizer,
        milestones=[config.warmup, config.epochs],
        warmup_iters=config.warmup,
        min_ratio=1e-8,
    )
    return optimizer, lr_scheduler


def apply_freeze_policy(model, config):
    if config.method in ("internvideo_c2c", "internvideo_c2c_1b"):
        model = build_lora(model, config)
        _apply_freeze_internvideo_c2c(model, config)
    elif config.method == "clip_aim":
        _apply_freeze_clip_aim(model, config)
    else:
        _apply_freeze_default(model, config)
    return model


def main():
    config = parser.parse_args()
    load_args(config.config, config)

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(local_rank)
    torch.cuda.set_device(local_rank)

    config.save_path = resolve_save_path(local_rank, config.save_path)

    log_file_path = None
    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(config.save_path, f'std_{timestamp}.out')
        os.makedirs(config.save_path, exist_ok=True)

    setup_for_distributed(local_rank == 0, log_file_path)
    print(config)
    set_seed(config.seed)

    print("training details")
    pprint.pprint(config)

    train_dataset, val_dataset, test_dataset = build_datasets(config)
    config.attrs = train_dataset.attrs
    if hasattr(train_dataset, "objs"):
        config.objs = train_dataset.objs

    model = build_model(config)

    if getattr(config, "pretrain_ckpt", False):
        load_pretrain(model, config.pretrain_ckpt)

    print("Setting gradient requirements")
    model = apply_freeze_policy(model, config)

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    optimizer, lr_scheduler = build_optimizer_and_scheduler(model, config)

    os.makedirs(config.save_path, exist_ok=True, mode=0o777)
    model = model.cuda(local_rank)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if any(p.requires_grad for p in model.parameters()):
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        print("Warning: No parameters require gradients. Skipping DDP wrapping to avoid crash.")

    if config.pretrain:
        msg = model.load_state_dict(torch.load(config.load_model)['state_dict'], strict=False)
        print("load custom ckpt : ", config.load_model)
        print(msg)

    if local_rank == 0:
        config_path = os.path.join(config.save_path, "config.yml")
        shutil.copyfile(config.config, config_path)

    if getattr(config, "use_cooc_penalty", False):
        print("USE train_comp_engine_penalty")
        train_comp_engine_penalty.train(
            model, optimizer, lr_scheduler, config,
            train_dataset, val_dataset, test_dataset, scaler,
        )
    else:
        print("USE train_comp_engine")
        train_comp_engine.train(
            model, optimizer, lr_scheduler, config,
            train_dataset, val_dataset, test_dataset, scaler,
        )

    cleanup()
    print("save_path : ", config.save_path)
    print("done!")


if __name__ == "__main__":
    main()
    exit(0)
