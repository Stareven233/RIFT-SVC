'''AdaMuon
from: https://github.com/Chongjie-Si/AdaMuon/blob/main/gpt2/models/optimizers/adamuon.py
似乎不能用于处理线性层wte,lm_head，需要跟Adam配合: https://github.com/Chongjie-Si/AdaMuon/blob/main/gpt2/models/model.py#L323
'''

import inspect

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn

from .chained_optimizer import ChainedOptimizer, OptimizerSpec


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
  assert G.ndim >= 2
  a, b, c = (3.4445, -4.7750, 2.0315)
  X = G.bfloat16()
  if G.size(-2) > G.size(-1):
    X = X.mT

  X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
  for _ in range(steps):
    A = X @ X.mT
    B = b*A + c*A@A
    X = a*X + B@X

  if G.size(-2) > G.size(-1):
    X = X.mT
  return X


class AdaMuon_Dist(Optimizer):

  def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, eps=1e-8, rank=0, world_size=1):
    if (rank is None) or (world_size is None):
      raise Exception('world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.')
    self.rank = rank
    self.world_size = world_size
    defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, eps=eps)
    params: list[Tensor] = [*params]
    param_groups = []
    for size in {p.numel() for p in params}:
      buf = torch.empty(world_size, size, dtype=torch.bfloat16, device='cuda')
      group = dict(params=[p for p in params if p.numel() == size], update_buffer=buf, update_buffer_views=[buf[i] for i in range(world_size)])
      param_groups.append(group)
    super().__init__(param_groups, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    for group in self.param_groups:
      update_buffer: Tensor = group['update_buffer']
      update_buffer_views: list[Tensor] = group['update_buffer_views']
      params: list[Tensor] = group['params']
      eps = group['eps']
      handle = None
      params_world = None

      def update_prev():
        if handle is not None:
          handle.wait()
        for p_world, g_world in zip(params_world, update_buffer_views):
          p_world.mul_(1 - group['lr'] * group['weight_decay'])
          p_world.add_(g_world.view_as(p_world), alpha=-group['lr'])

      for base_i in range(len(params))[::self.world_size]:
        if base_i + self.rank < len(params):
          p = params[base_i + self.rank]
          g = p.grad
          assert g is not None

          state = self.state[p]
          if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(g)

          buf: Tensor = state['momentum_buffer']
          buf.mul_(group['momentum']).add_(g)

          g = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
          if g.ndim == 4:
            g = g.view(len(g), -1)
          g = zeropower_via_newtonschulz5(torch.sign(g), steps=group['ns_steps']).flatten()

          if 'v_buffer' not in state:
            state['v_buffer'] = torch.zeros_like(g)
          v = state['v_buffer']
          v.mul_(group['momentum']).addcmul_(g, g, value=1 - group['momentum'])

          g = g.div(v.view_as(g).sqrt().add(eps))
          scale = 0.2 * (min(p.shape) * max(p.shape))**0.5 / (g.norm() + eps)
          g.mul_(scale)
          g = g.to(update_buffer.dtype)

        else:
          g = update_buffer_views[self.rank]
        if base_i > 0:
          update_prev()
        # handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
          handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
        else:
          # 单卡直接复制
          update_buffer.copy_(g)
        params_world = params[base_i:base_i + self.world_size]
      update_prev()


def get_params_for_adamuon(model):
  '''
    Split parameters into those that can be optimized by AdaMuon (2-D, non-Embedding)
    and those that should use AdamW.

    Args:
        model: The model to scan.

    Returns:
        (adamuon_params, adamw_params): Two lists of unique parameters.
    '''
  adamuon_params, adamw_params = set(), set()

  for module in model.modules():
    for param in module.parameters(recurse=False):
      if not param.requires_grad:
        continue
      # AdaMuon适用于2D参数，排除嵌入层
      if isinstance(module, nn.Embedding) or param.ndim < 2:
        adamw_params.add(param)
      else:
        adamuon_params.add(param)

  # 转换为list并保持确定性顺序
  adamuon_params = sorted(adamuon_params, key=lambda p: p.data_ptr())
  adamw_params = sorted(adamw_params, key=lambda p: p.data_ptr())
  print(f'sort {len(adamuon_params)} params for adamuon, and {len(adamw_params)} for adamw')
  return adamuon_params, adamw_params


class AdaMuon(Optimizer):
  '''
    AdaMuon2 - Combines AdaMuon and AdamW in a single optimizer
    Automatically selects the appropriate optimization method for different parameters.
    
    https://github.com/Chongjie-Si/AdaMuon
    
    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    
    Arguments:
        lr: The learning rate. (0.02 is a good default)
        weight_decay: Weight decay (default: 0.01)
        adamuon_params: The parameters to be optimized by AdaMuon.
        momentum: The momentum used by the internal AdaMuon. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal AdaMuon. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `adamuon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW. (default: (0.9, 0.95))
        adamw_eps: The epsilon for the internal AdamW. (default: 1e-8)
        optim_groups: Custom parameter groups for advanced usage.
    '''

  def __init__(
      self,
      lr=0.02,
      weight_decay=0.01,
      adamuon_params=None,
      momentum=0.95,
      nesterov=True,
      ns_steps=5,
      adamw_params=None,
      adamw_betas=(0.9, 0.95),
      adamw_eps=1e-8,
      optim_groups=None,
  ):
    defaults = dict(
      lr=lr,
      weight_decay=weight_decay,
      momentum=momentum,
      nesterov=nesterov,
      ns_steps=ns_steps,
      eps=1e-8,
      adamw_betas=adamw_betas,
      adamw_eps=adamw_eps,
    )

    # adamuon_params, adamw_params 必须得有，且二者之和与optim_groups所包含的参数一致，但实际用于初始化的是optim_groups
    if optim_groups is None:
      params = list(adamuon_params)
      params.extend(adamw_params)
    else:
      params = optim_groups

    super().__init__(params, defaults)

    # 为每个参数设置优化方法标记
    for p in adamuon_params:
      self.state[p]['use_adamuon'] = True
    for p in adamw_params:
      self.state[p]['use_adamuon'] = False

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      lr = group['lr']
      weight_decay = group['weight_decay']

      ############################
      #          AdaMuon         #
      ############################

      params = [p for p in group['params'] if self.state[p]['use_adamuon']]
      for p in params:
        g = p.grad
        if g is None:
          continue

        state = self.state[p]

        # 初始化动量缓冲区
        if 'momentum_buffer' not in state:
          state['momentum_buffer'] = torch.zeros_like(g)
        buf = state['momentum_buffer']
        buf.mul_(group['momentum']).add_(g)

        # Nesterov动量
        g = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf

        # 对卷积权重展平处理
        orig_shape = g.shape
        if g.ndim == 4:
          g = g.view(len(g), -1)

        # Newton-Schulz归一化
        g = zeropower_via_newtonschulz5(torch.sign(g), steps=group['ns_steps']).flatten()

        # 自适应方差缓冲区
        if 'v_buffer' not in state:
          state['v_buffer'] = torch.zeros_like(g)
        v = state['v_buffer']
        v.mul_(group['momentum']).addcmul_(g, g, value=1 - group['momentum'])

        # 自适应缩放
        g = g.div(v.view_as(g).sqrt().add(group['eps']))
        scale = 0.2 * (min(p.shape) * max(p.shape))**0.5 / (g.norm() + group['eps'])
        g.mul_(scale)
        # 恢复原始形状
        g = g.view(orig_shape)

        # 权重衰减 + 参数更新
        p.mul_(1 - group['lr'] * group['weight_decay'])
        p.add_(g, alpha=-group['lr'])

      ############################
      #          AdamW           #
      ############################

      params = [p for p in group['params'] if not self.state[p]['use_adamuon']]
      beta1, beta2 = group['adamw_betas']
      adamw_eps = group['adamw_eps']

      for p in params:
        g = p.grad
        if g is None:
          continue

        state = self.state[p]

        # 初始化AdamW状态
        if 'step' not in state:
          state['step'] = 0
          state['exp_avg'] = torch.zeros_like(g)
          state['exp_avg_sq'] = torch.zeros_like(g)

        state['step'] += 1

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        # 更新一阶动量
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)

        # 更新二阶动量
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        # 偏差校正
        bias_correction1 = 1 - beta1**state['step']
        bias_correction2 = 1 - beta2**state['step']

        # 计算AdamW更新
        denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(adamw_eps)
        step_size = lr / bias_correction1

        # 权重衰减 + 参数更新
        p.mul_(1 - lr*weight_decay)
        p.addcdiv_(exp_avg, denom, value=-step_size)
  
    return loss


class AdaMuonOld(Optimizer):

  def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, eps=1e-8):
    defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, eps=eps)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    for group in self.param_groups:
      params = group['params']
      eps = group['eps']

      for p in params:
        g = p.grad
        if g is None:
          continue

        state = self.state[p]

        # 初始化动量缓冲区
        if 'momentum_buffer' not in state:
          state['momentum_buffer'] = torch.zeros_like(g)
        buf = state['momentum_buffer']
        buf.mul_(group['momentum']).add_(g)

        # Nesterov 动量
        g = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
        # 对卷积权重展平处理（保持原逻辑）
        orig_shape = g.shape
        if g.ndim == 4:
          g = g.view(len(g), -1)
        # Newton-Schulz 归一化（保持原函数）
        g = zeropower_via_newtonschulz5(torch.sign(g), steps=group['ns_steps']).flatten()

        # 自适应方差缓冲区
        if 'v_buffer' not in state:
          state['v_buffer'] = torch.zeros_like(g)
        v = state['v_buffer']
        v.mul_(group['momentum']).addcmul_(g, g, value=1 - group['momentum'])

        # 自适应缩放
        g = g.div(v.view_as(g).sqrt().add(eps))
        scale = 0.2 * (min(p.shape) * max(p.shape))**0.5 / (g.norm() + eps)
        g.mul_(scale)

        # 恢复原始形状（如果是卷积）
        g = g.view(orig_shape)
        # 权重衰减 + 参数更新
        p.mul_(1 - group['lr'] * group['weight_decay'])
        p.add_(g, alpha=-group['lr'])


class AdaMuonWrapper(ChainedOptimizer):

  def __init__(self, model, lr, betas, weight_decay=0.0, rank=0, world_size=1):
    adam_groups, adamuon_params = self.__configure_optimizers(model, weight_decay)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    optimizer_adamw = torch.optim.AdamW(adam_groups, lr=lr, betas=betas, fused=fused_available)
    # optimizer_adamuon = AdaMuonOld(adamuon_params, lr=lr, momentum=0.95, rank=rank, world_size=world_size, weight_decay=weight_decay)
    optimizer_adamuon = AdaMuonOld(adamuon_params, lr=lr, momentum=0.95, weight_decay=weight_decay)

    adamuon_params_id_set = set(id(p) for p in adamuon_params)
    spec_adamw = OptimizerSpec(torch.optim.AdamW, None, None)
    spec_adamuon = OptimizerSpec(AdaMuonOld, None, lambda param: id(param) in adamuon_params_id_set)
    optims = [optimizer_adamw, optimizer_adamuon]
    specs = [spec_adamw, spec_adamuon]
    super().__init__(optims, specs)

  @staticmethod
  def __configure_optimizers(model, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    nodecay_params, twoD_params, decay_params = [], [], []
    for n, p in param_dict.items():
      # 1. 首先检查是否（维度 < 2）
      if p.dim() < 2:
        nodecay_params.append(p)
      # 2. 如果不，再检查是否是特殊的权重（需要weight decay）
      elif 'wte' in n or 'lm_head' in n:
        decay_params.append(p)
      # 3. 这里的 p.dim() 必然 >= 2，且名称不包含 'wte' 或 'lm_head'
      else:
        twoD_params.append(p)

    adam_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
    print(f'num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')

    num_twoD_params = sum(p.numel() for p in twoD_params)
    print(f'num 2D parameter tensors: {len(twoD_params)}, with {num_twoD_params:,} parameters')

    return adam_groups, twoD_params
