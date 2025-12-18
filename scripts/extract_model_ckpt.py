'''
从 PyTorch checkpoint 中提取 model 和 hyper_parameters 相关的权重，
去掉优化器等其他信息后保存为新的 .ckpt 文件
'''

import torch
from pathlib import Path
from typing import Union, Optional


def extract_model_checkpoint(
    state_dict: dict,
    output_path: Union[str, Path],
    keys: set[str] = {'model', 'hyper_parameters'},
) -> dict:
  '''
    从 torch.load 返回的 state_dict 中提取指定前缀的键值对，保存为精简的 checkpoint 文件。
    
    Args:
        state_dict: torch.load() 返回的字典
        output_path: 输出文件路径，如果为 None 则不保存文件
        keys: 要保留的键，默认为 ('model', 'hyper_parameters')
    
    Returns:
        提取后的新 state_dict
    '''
  new_state_dict = dict()

  for key, value in state_dict.items():
    if key in keys:
      new_state_dict[key] = value

  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(new_state_dict, output_path)
  print(f'已保存精简 checkpoint 到: {output_path}')
  print(f'原始键数量: {len(state_dict)}, 提取后键数量: {len(new_state_dict)}')

  return new_state_dict


def extract_and_save(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]],
    keep_prefixes: tuple[str, ...] = None,
    drop_prefixes: tuple[str, ...] = (),
) -> dict:
  '''
    从 checkpoint 文件中提取指定前缀的键值对并保存。
    
    Args:
        input_path: 输入 checkpoint 文件路径
        output_path: 输出文件路径，如果为 None 则自动生成（添加 _extracted 后缀）
        keep_prefixes: 要保留的键前缀，一般为 ('model', 'hyper_parameters')
        drop_prefixes: 要丢弃的键前缀，一般为 ('optimizer_states')，优先级更高
    
    Returns:
        提取后的新 state_dict
    '''
  input_path = Path(input_path)
  print(f'加载 checkpoint: {input_path}')
  state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
  
  final_keys = set()
  for key in state_dict.keys():
    if any(key.startswith(prefix) for prefix in drop_prefixes):
      continue
    if keep_prefixes is None or any(key.startswith(prefix) for prefix in keep_prefixes):
      final_keys.add(key)

  if output_path is None:
    output_path = input_path.with_stem(f'extracted-{input_path.stem}')

  extract_model_checkpoint(state_dict, output_path, final_keys)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='从 checkpoint 中中提取或过滤键')
  parser.add_argument('input', type=str, help='输入 checkpoint 文件路径')
  parser.add_argument('-o', '--output', type=str, default=None, help='输出文件路径')
  parser.add_argument('-kp', '--keep_prefixes', type=str, nargs='+', required=False, help='要保留的键前缀')
  parser.add_argument('-dp', '--drop_prefixes', type=str, nargs='+', default=('optimizer_states', ), help='要丢弃的键前缀')
  # filter_group = parser.add_mutually_exclusive_group(required=True)
  # # 添加互斥的参数
  # filter_group.add_argument('-kp', '--keep_prefixes', type=str, nargs='+', help='要保留的键前缀（与 drop_prefixes 互斥）')
  # filter_group.add_argument('-dp', '--drop_prefixes', type=str, nargs='+', help='要丢弃的键前缀（与 keep_prefixes 互斥）')

  args = parser.parse_args()
  extract_and_save(args.input, args.output, args.keep_prefixes, args.drop_prefixes)
