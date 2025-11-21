import math
from collections import defaultdict

from schedulefree import AdamWScheduleFree
from torch.optim import AdamW

from .muon_moonshot import get_params_for_muon, Muon
from .adamuon import AdaMuonWrapper
from . import lr_scheduler


def divide_optim_groups(model, lr, weight_decay, lora_training=False):
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
                    specp_decay_params[f'specp_decay_{fan_ratio:.2f}'].append(p)
                    specp_decay_lr[f'specp_decay_{fan_ratio:.2f}'] = lr * fan_ratio
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
    return optim_groups


def get_optimizer(optimizer_type, model, lr, betas, weight_decay, lora_training=False, **kwargs):
    optim_groups = divide_optim_groups(model, lr, weight_decay, lora_training)
    max_steps = kwargs['max_steps']
    warmup_ratio = kwargs.get('warmup_ratio', 0.05)
    warmup_steps = int(max_steps * warmup_ratio)
    decay_step = kwargs.get('decay_step')
    decay_rate = kwargs.get('decay_rate', 0.5)
    anneal_ratio = kwargs.get('anneal_ratio', 0.2)
    global_step = kwargs.get('global_step', -1)
    if global_step != -1:
        # resuming an optimizer
        for g in optim_groups:
            g['initial_lr'] = g['lr']

    match optimizer_type:
        case 'adamwsf':
            optimizer = AdamWScheduleFree(optim_groups, betas=betas, warmup_steps=warmup_steps)
            scheduler = None
        case 'adamw':
            optimizer = AdamW(optim_groups, betas=betas, weight_decay=weight_decay)
            min_lr = kwargs.get('min_lr', 0.0)
            scheduler = lr_scheduler.LinearWarmupDecayLR(optimizer, warmup_steps, max_steps, min_lr=min_lr)
        case 'muon':
            pm, po = get_params_for_muon(model)
            # 笨办法兼容optim_groups
            optimizer = Muon(lr, weight_decay, pm, adamw_params=po, optim_groups=optim_groups)
            scheduler = lr_scheduler.warmup_decay_anneal(optimizer, max_steps, warmup_ratio, decay_step, decay_rate, anneal_ratio, last_steps=global_step)
        case 'adamuon' if not lora_training:
            optimizer = AdaMuonWrapper(model, lr, betas, weight_decay, rank=0, world_size=1)
            if global_step != -1:
                for g in optimizer.param_groups:
                    g['initial_lr'] = g['lr']
            scheduler = lr_scheduler.warmup_decay_anneal(optimizer, max_steps, warmup_ratio, decay_step, decay_rate, anneal_ratio, last_steps=global_step)
        case _:
            raise ValueError(f'Invalid optimizer type: {optimizer_type} with {lora_training=}')

    return optimizer, scheduler
