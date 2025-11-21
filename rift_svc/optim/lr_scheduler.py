import math
from functools import partial

from torch.optim import Optimizer
from torch.optim import lr_scheduler


# step scheduler
def fn_linear_warmup(warmup_steps, step):
  if step < warmup_steps:  # linear warmup
    return float(step) / float(max(1, warmup_steps))
  else:
    return 1.0


def linear_warmup(optimizer: Optimizer, warmup_steps):
  # return partial(fn_linear_warmup, warmup_steps)
  scheduler = lr_scheduler.LambdaLR(optimizer, partial(fn_linear_warmup, warmup_steps))
  return scheduler


def fn_linear_warmup_cosine_decay(warmup_steps, max_steps, multipler_min, step):
  if step < warmup_steps:  # linear warmup
    return float(step) / float(max(1, warmup_steps))
  else:  # cosine learning rate schedule
    multipler = 0.5 * (math.cos((step-warmup_steps) / (max_steps-warmup_steps) * math.pi) + 1)
    return max(multipler, multipler_min)


def linear_warmup_cosine_decay(optimizer: Optimizer, warmup_steps, max_steps, multipler_min):
  # return partial(fn_linear_warmup_cosine_decay, warmup_steps, max_steps, multipler_min)
  scheduler = lr_scheduler.LambdaLR(optimizer, partial(fn_linear_warmup_cosine_decay, warmup_steps, max_steps, multipler_min))
  return scheduler


def linear_warmup_drop(optimizer: Optimizer, steps_per_epoch: int, warmup_epochs: int, drop_epoch_list: None | tuple[int], drop_rate: float = 0.1):
  # if drop_epochs is not None:
  #   drop_steps = tuple()
  warmup_steps = max(1, int(steps_per_epoch * warmup_epochs))
  rate = 1.0

  def inner(step):
    nonlocal rate
    nonlocal drop_epoch_list
    if step < warmup_steps:  # linear warmup
      return step / warmup_steps
    elif not isinstance(drop_epoch_list, (list, tuple)) or len(drop_epoch_list) == 0:
      return rate
    i = 0
    for e in drop_epoch_list:
      if step > e * steps_per_epoch:
        i += 1
      else:
        break
    rate = rate * (drop_rate**i)
    drop_epoch_list = drop_epoch_list[i:]
    return rate

  scheduler = lr_scheduler.LambdaLR(optimizer, inner)
  return scheduler


def linear_warmup_decay(optimizer: Optimizer, warmup_steps: int, decay_per_steps: int, decay_rate: float = 0.1, last_steps=-1):
  warmup_steps = max(0, min(warmup_steps, decay_per_steps))
  rate = 1.0
  last_decay_step = warmup_steps
  if last_steps > warmup_steps:
    n = (last_steps-warmup_steps) // decay_per_steps
    rate = decay_rate**n
    last_decay_step += decay_per_steps * n

  def inner(step):
    nonlocal rate
    nonlocal last_decay_step
    if step < warmup_steps:  # linear warmup
      return step / warmup_steps
    elif (step - last_decay_step) >= decay_per_steps:
      last_decay_step = step
      rate *= decay_rate
    return rate

  scheduler = lr_scheduler.LambdaLR(optimizer, inner, last_steps)
  return scheduler


def cosine_annealing(optimizer: Optimizer, max_lr: float, max_steps: int, warmup_ratio=0, final_lr_ratio=1e-4, **kwargs):
  div_factor = kwargs.get('div_factor', 25)
  final_div_factor = 1 / (div_factor*final_lr_ratio)
  scheduler = lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=max_lr,
      total_steps=max_steps,
      pct_start=warmup_ratio,  # 从 initial_lr warmup 到 max_lr 所占的step比例
      div_factor=div_factor,  # initial_lr = max_lr/div_factor
      final_div_factor=final_div_factor,  # 余弦退火最终学习率相比initial_lr降低的倍数，div_factor默认25，因此分子就是相比max_lr降低的倍数
      # **kwargs,
  )
  return scheduler


def warmup_stable_decay(optimizer: Optimizer, max_steps: int, warmup_ratio=0, decay_ratio=0.2, **_):
  # WSD策略（Warmup-Stable-Decay） @Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
  n_warmup = max_steps * warmup_ratio
  n_decay = max_steps * decay_ratio  # n大于20k，可小于0.2

  def inner(step):
    if step < n_warmup:
      return step / n_warmup  # 线性增长
    elif step <= max_steps - n_decay:
      return 1  # 稳定
    else:
      t = (step - (max_steps-n_decay)) / n_decay  # (0 -> 1)
      t = 1 - t**0.5  # (1 -> 0)
      return t

  scheduler = lr_scheduler.LambdaLR(optimizer, inner)
  return scheduler

def warmup_decay_anneal(optimizer: Optimizer, max_steps: int, warmup_ratio=0.05, decay_step: int|list[int]|None=None, decay_rate=0.5, anneal_ratio=0.2, last_step=-1):
  '''带阶段式学习率衰减的WSD调度   
  学习率缓慢上升_学习率阶段性下降_学习率退火  
  decay_step: int(固定步数下调学习率) / list(列出该下调学习率的step) / None(去掉阶段性下降阶段)
  '''

  # 学习率按decay_step固定值、指定值分为两种具体 decay 类型
  fixed_step = isinstance(decay_step, int)
  if decay_step is None or not fixed_step and len(decay_step) == 0:
    decay_step = max_steps + 1
  elif not fixed_step:
    decay_step = sorted(decay_step)
  n_warmup = max_steps * warmup_ratio
  n_warmup = max(0, min(n_warmup, decay_step if fixed_step else decay_step[0]))
  n_anneal = max_steps * anneal_ratio
  anneal_step = max_steps - n_anneal
  rate = 1.0
  # 恢复训练时根据当前步数重置rate
  last_decay_step = n_warmup
  if last_step > n_warmup:
    if fixed_step:
      n = (last_step-n_warmup) // decay_step
      last_decay_step += decay_step * n
    else:
      n = 0
      for d in decay_step:
        if last_step < d:
          break
        n += 1
      decay_step = decay_step[n:]
    rate = decay_rate**n

  def inner(step):
    nonlocal rate
    nonlocal last_decay_step
    # # linear warmup
    if step < n_warmup:
      return step / n_warmup
    # annealing
    elif step >= anneal_step:
      t = (step-anneal_step) / n_anneal  # (0 -> 1)
      t = 1 - t**0.5  # (1 -> 0)
      return rate * t
    # decay
    elif fixed_step and (step - last_decay_step) >= decay_step:
      rate *= decay_rate
      last_decay_step = step
    elif step > decay_step[0]:
      rate *= decay_rate
      decay_step.pop(0)
    return rate

  scheduler = lr_scheduler.LambdaLR(optimizer, inner, last_step)
  return scheduler


class LinearWarmupDecayLR(lr_scheduler._LRScheduler):
  """
    Linear learning rate scheduler with warmup and minimum lr.
    
    During warmup, the LR increases linearly from 0 to the base LR.
    After warmup, the LR decays linearly from the base LR down to min_lr.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps to linearly increase LR.
        total_steps (int): Total number of steps for training (warmup + decay).
        min_lr (float): Minimum learning rate after decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

  def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
    if total_steps <= warmup_steps:
      raise ValueError("Total steps must be larger than warmup_steps for decay to happen.")
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.min_lr = min_lr
    super(LinearWarmupDecayLR, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    """Compute learning rate using linear warmup and then linear decay."""
    # Note: self.last_epoch is incremented by the base _LRScheduler.step() before calling get_lr().
    if self.last_epoch < self.warmup_steps:
      # Warmup phase: increase linearly from 0 (or a small value) to base_lr.
      return [base_lr * float(self.last_epoch + 1) / float(self.warmup_steps) for base_lr in self.base_lrs]
    else:
      # Decay phase: decrease linearly from base_lr to min_lr.
      progress = float(self.last_epoch - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
      return [max(base_lr * (1.0-progress) + self.min_lr * progress, self.min_lr) for base_lr in self.base_lrs]
