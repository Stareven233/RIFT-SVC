import json
from pathlib import Path
import json
import random
from functools import partial
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from rift_svc.utils import linear_interpolate_tensor


class LazyTensorLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self._tensor = None

    @property
    def tensor(self):
        if self._tensor is None:
            self._tensor = torch.load(self.file_path, mmap=False)
        return self._tensor


# pt_load = partial(torch.load, weights_only=True, map_location='cpu', mmap=True)
def pt_load(path, key, loc='cpu', mmap=False):
    # 小文件没必要开mmap
    p = path.with_suffix(f'.{key}.pt')
    return torch.load(p, weights_only=True, map_location=loc, mmap=mmap).squeeze(0)


class SVCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_info_path: str,
        max_frame_len = 256,
        split = "train",
        use_cvec_downsampled: bool = False,
        cvec_downsample_rate: int = 2,
        n_samples = None,
        lazy = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_frame_len = max_frame_len

        with open(meta_info_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        speakers = meta["speakers"]
        self.num_speakers = len(speakers)
        self.spk2idx = {spk: idx for idx, spk in enumerate(speakers)}
        self.split = split
        self.samples = meta[f"{split}_audios"][:n_samples]
        self.use_cvec_downsampled = use_cvec_downsampled
        self.cvec_downsample_rate = cvec_downsample_rate
        self.cache = self._load_cache_lazy(lazy)

    def _load_cache_lazy(self, lazy=True):
        cache = defaultdict(list)
        for s in tqdm(self.samples, desc='loading preprocessed cache'):
            spk = s['speaker']
            path = self.data_dir / spk / s['file_name']
            mel = pt_load(path, 'mel').T
            cache['spk_id'].append(None if lazy else torch.LongTensor([self.spk2idx[spk]]))
            cache['f0'].append(None if lazy else pt_load(path, 'f0'))
            cache['rms'].append(None if lazy else pt_load(path, 'rms'))
            cache['cvec'].append(None if lazy else pt_load(path, 'cvec'))
            cache['mel'].append(mel)
            # 采样权重，长度小于 max_frame_len 的均是同等的一次采样
            cache['weight'].append(max(self.max_frame_len, mel.shape[0]) ** 0.6)
        return cache

    def get_frame_len(self, index):
        return self.cache['mel'][index].shape[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        def load(key, func):
            if (data := self.cache[key][index]) is not None:
                return data
            data = func()
            self.cache[key][index] = data
            return data

        sample = self.samples[index]
        spk = sample['speaker']
        path = self.data_dir / spk / sample['file_name']

        spk_id = load('spk_id', lambda: torch.LongTensor([self.spk2idx[spk]]))  # [1]
        mel = load('mel', lambda: pt_load(path, 'mel').T)
        rms = load('rms', lambda: pt_load(path, 'rms'))
        f0 = load('f0', lambda: pt_load(path, 'f0'))
        cvec = load('cvec', lambda: pt_load(path, 'cvec'))

        frame_len = mel.shape[0]
        cvec = linear_interpolate_tensor(cvec, frame_len)
        if self.use_cvec_downsampled:
            cvec_ds = cvec[::2, :]
            cvec_ds = linear_interpolate_tensor(cvec_ds, cvec_ds.shape[0]//self.cvec_downsample_rate)
            cvec_ds = linear_interpolate_tensor(cvec_ds, frame_len)

        if frame_len > self.max_frame_len:
            if self.split == "train": 
                # Keep trying until we find a good segment or hit max attempts
                max_attempts = 10
                attempt = 0
                while attempt < max_attempts:
                    start = random.randint(0, frame_len - self.max_frame_len)
                    end = start + self.max_frame_len
                    f0_segment = f0[start:end]
                    # Check if more than 90% of f0 values are 0
                    zero_ratio = (f0_segment == 0).float().mean().item()
                    if zero_ratio < 0.9:  # Found a good segment
                        break
                    attempt += 1
            else:
                start = 0
            end = start + self.max_frame_len
            mel = mel[start:end]
            rms = rms[start:end]
            f0 = f0[start:end]
            cvec = cvec[start:end]
            if self.use_cvec_downsampled:
                cvec_ds = cvec_ds[start:end]
            frame_len = self.max_frame_len

        result = dict(
            spk_id = spk_id,
            mel = mel,
            rms = rms,
            f0 = f0,
            cvec = cvec,
            frame_len = frame_len
        )

        if self.use_cvec_downsampled:
            result['cvec_ds'] = cvec_ds

        return result


def collate_fn(batch):
    spk_ids, mels, rmss, f0s, cvecs, frame_lens = [], [], [], [], [], []
    for item in batch:
        spk_ids.append(item['spk_id'])
        mels.append(item['mel'])
        rmss.append(item['rms'])
        f0s.append(item['f0'])
        cvecs.append(item['cvec'])
        frame_lens.append(item['frame_len'])

    # Pad sequences to max length
    mels_padded = pad_sequence(mels, batch_first=True)
    rmss_padded = pad_sequence(rmss, batch_first=True)
    f0s_padded = pad_sequence(f0s, batch_first=True)
    cvecs_padded = pad_sequence(cvecs, batch_first=True)

    if 'cvec_ds' in batch[0]:
        cvecs_ds = [item['cvec_ds'] for item in batch]
        cvecs_ds_padded = pad_sequence(cvecs_ds, batch_first=True)

    spk_ids = torch.cat(spk_ids)
    frame_len = torch.tensor(frame_lens)

    result = {
        'spk_id': spk_ids,
        'mel': mels_padded,
        'rms': rmss_padded,
        'f0': f0s_padded,
        'cvec': cvecs_padded,
        'frame_len': frame_len
    }

    if 'cvec_ds' in batch[0]:
        result['cvec_ds'] = cvecs_ds_padded

    return result
