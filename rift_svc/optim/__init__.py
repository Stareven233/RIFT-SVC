import math
from collections import defaultdict

from schedulefree import AdamWScheduleFree
from torch.optim import AdamW

from .muon_moonshot import get_params_for_muon, Muon
from . import lr_scheduler


def get_optimizer(optimizer_type, model, lr, betas, weight_decay, warmup_steps, lora_training=False, **kwargs):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    if not lora_training:
        specp_decay_params = defaultdict(list)
        specp_decay_lr = {}
        decay_params = []
        nodecay_params = []
        for n, p in param_dict.items():
            if p.dim() >= 2:
                if n.endswith('out.weight') or n.endswith('proj.weight'):
                    fan_out, fan_in = p.shape[-2:]
                    fan_ratio = fan_out / fan_in
                    specp_decay_params[f"specp_decay_{fan_ratio:.2f}"].append(p)
                    specp_decay_lr[f"specp_decay_{fan_ratio:.2f}"] = lr * fan_ratio
                else:
                    decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr}
        ] + [
            {'params': params, 'weight_decay': weight_decay, 'lr': specp_decay_lr[group_name]}
            for group_name, params in specp_decay_params.items()
        ]
    else:
        lora_a_or_spk_embed_params = []
        lora_b_params = []
        for n, p in param_dict.items():
            if n.endswith('.A.weight') or n.endswith('.spk_embed.weight'):
                lora_a_or_spk_embed_params.append(p)
            elif n.endswith('.B.weight'):
                lora_b_params.append(p)
        dim = model.transformer.dim
        rank = model.transformer.transformer_blocks[0].attn.k_proj.rank
        optim_groups = [
            {'params': lora_a_or_spk_embed_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': lora_b_params, 'weight_decay': weight_decay, 'lr': lr*math.sqrt(dim/rank)}
        ]
    
    if kwargs.get('global_step', -1) != -1:
        # resuming an optimizer
        for g in optim_groups:
            g['initial_lr'] = g['lr']

    if optimizer_type == 'adamwsf':
        optimizer = AdamWScheduleFree(optim_groups, betas=betas, warmup_steps=warmup_steps)
        return optimizer, None
    elif optimizer_type == 'adamw':
        optimizer = AdamW(optim_groups, betas=betas, weight_decay=weight_decay)
        max_steps = kwargs['max_steps']
        min_lr = kwargs.get('min_lr', 0.0)
        scheduler = lr_scheduler.LinearWarmupDecayLR(optimizer, warmup_steps, max_steps, min_lr=min_lr)
        return optimizer, scheduler
    elif optimizer_type == 'moun':
        pm, po = get_params_for_muon(model)
        # 笨办法兼容optim_groups
        warmup_ratio = warmup_steps / kwargs['max_steps']
        optimizer = Muon(lr, weight_decay, pm, adamw_params=po, optim_groups=optim_groups)
        # scheduler = lr_scheduler.cosine_annealing(optimizer, lr, kwargs['max_steps'], warmup_ratio, decay_rate=kwargs['gamma'])
        scheduler = lr_scheduler.warmup_stage_decay(optimizer, kwargs['decay_step'], kwargs['max_steps'], warmup_ratio=warmup_ratio, decay_ratio=0.1, decay_rate=kwargs['gamma'], last_steps=kwargs['global_step'])
        return optimizer, scheduler
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")
