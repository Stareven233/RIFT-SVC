r'''
https://github.com/Pur1zumu/RIFT-SVC
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
cd D:\Code\projects\RIFT-SVC
uv add pyworld
uv add numpy==2.2.6
New-Item -ItemType SymbolicLink -Path "D:\Code\projects\RIFT-SVC\pretrained\rmvpe\model.pt" -Target "D:\Code\projects\DDSP-SVC\pretrain\rmvpe\model.pt"
New-Item -ItemType SymbolicLink -Path "D:\Code\projects\RIFT-SVC\pretrained\vocoder" -Target "D:\Code\projects\DDSP-SVC\pretrain\vocoder"

!这个项目不需要提前切片，会在训练时随机借助python序列的slice功能切

cd D:\Code\projects\RIFT-SVC
$DATA_DIR="data/megumin"
$DATA_DIR="data/fritia"
$DATA_DIR="D:\Code\projects\RIFT-SVC\data\fritia\t"
uv run scripts/resample_normalize_audios.py --src $DATA_DIR
uv run scripts/prepare_data_meta.py --data-dir $DATA_DIR --num-test 20
uv run scripts/prepare_mel.py --data-dir $DATA_DIR --num-workers 0
uv run scripts/prepare_rms.py --data-dir $DATA_DIR --num-workers 0
uv run scripts/prepare_f0.py --data-dir $DATA_DIR --num-workers 0
uv run scripts/prepare_cvec.py --data-dir $DATA_DIR --num-workers 0

cd D:\Code\projects\RIFT-SVC
$overrides = @("training.run_name=test","dataset.n_samples=20","training.max_steps=23","training.batch_size_per_gpu=2")
$name = "fritia"
$overrides = @("training.run_name=${name}-r4","training.max_steps=2410","training.decay_step=600","training.test_per_steps=800")
$name = "megumin"
$overrides = @("training.run_name=${name}-xpred-r2","training.max_steps=5200","training.decay_step=1500","training.test_per_steps=1000")
$overrides = @("training.run_name=${name}-xpred","training.learning_rate=5e-5","training.max_steps=20200","training.decay_step=29000","training.test_per_steps=1000")

uv run train.py name=$name @overrides
uv run train.py name=$name @overrides training.resume_from_checkpoint=ckpts/${name}-xpred/model-step\=4000.ckpt
uv run train.py name=$name @overrides training.pretrained_path=ckpts/${name}-r2/model-step\=1059.ckpt
uv run train.py name=$name training.freeze_adaln_and_tembed=false training.drop_spk_prob=0.2 training.pretrained_path=pretrained/pretrain-v3_dit-768-12.ckpt

Write-Host "等待10分钟..."
Start-Sleep -Seconds 600
tensorboard --logdir D:/Code/projects/RIFT-SVC/logs
cd D:\Code\projects\RIFT-SVC

#todo
lambda学习率调度
Muon改为真正继承Optimizer
改造配置读取方式，模糊掉 name 的输入
lazy cache, 每个样本读取时完整加载
dataset转为iter类型，或者多倍，减少epoch交替时速度降低
音区偏移
'''

from pathlib import Path
import hydra
from lightning import Trainer
from lightning.pytorch import seed_everything
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from lightning.pytorch import LightningModule

from rift_svc import DiT, RF
from rift_svc.dataset import SVCDataset, collate_fn
from rift_svc.lightning_module import RIFTSVCLightningModule
from rift_svc.optim import get_optimizer
from rift_svc.utils import load_state_dict
from rift_svc.utils import CustomProgressBar, ModelCheckpoint2, EnsureFinalValidationCallback
from rift_svc.utils import ckpt_step_patten
from rift_svc.utils import safe_save_hyperparameters

LightningModule.save_hyperparameters = safe_save_hyperparameters
torch.set_float32_matmul_precision('high')
# from omegaconf.base import ContainerMetadata
# import typing
# from collections import defaultdict
# torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, typing.Any, dict, defaultdict])


@hydra.main(version_base=None, config_path='config', config_name='noe')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    train_dataset = SVCDataset(**cfg.dataset, split='train')
    val_dataset = SVCDataset(**cfg.dataset, split='test')
    transformer = DiT(
        **cfg.model,
        num_speaker=train_dataset.num_speakers,
    )
    rf = RF(
        transformer=transformer,
        time_schedule=cfg.training.time_schedule,
    )

    # Load pretrained weights if specified
    resume_ckpt = cfg.training.get('resume_from_checkpoint', None)
    pre_ckpt = cfg.training.get('pretrained_path', None)
    if resume_ckpt is None and pre_ckpt is not None:
        state_dict = torch.load(pre_ckpt, map_location='cuda', weights_only=False)
        # print(f'{state_dict.keys()=}')  # (['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'hparams_name', 'hyper_parameters'])
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Load only model weights, allowing mismatched keys for speaker embeddings
        missing_keys, unexpected_keys = load_state_dict(rf, state_dict)
        print(f"Loaded pretrained model from {pre_ckpt}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    if cfg.training.get('lora_training', False):
        rf.transformer.apply_lora(cfg.training.lora_rank, cfg.training.lora_alpha)
    
    if cfg.training.get('freeze_adaln_and_tembed', True):
        rf.transformer.freeze_adaln_and_tembed()

    global_step = -1
    if resume_ckpt or pre_ckpt:
        m = ckpt_step_patten.search(resume_ckpt or pre_ckpt)
        global_step = (m and int(m.group(0))) or -1
    optimizer, lr_scheduler = get_optimizer(
        rf,
        cfg.training.optimizer_type,
        cfg.training,
        global_step=global_step,
        lora_training=cfg.training.get('lora_training', False),
    )
    OmegaConf.update(cfg, 'spk2idx', train_dataset.spk2idx, force_add=True)
    model = RIFTSVCLightningModule(
        model=rf,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        cfg=cfg
    )

    run_name = cfg.training.run_name
    ckpt_dir = Path('ckpts', run_name)
    ckpt_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, ckpt_dir / 'config.yaml', resolve=True)
    checkpoint_callback = ModelCheckpoint2(
        dirpath=ckpt_dir,
        filename='model-{step}',
        # monitor='val/si_snr',
        # mode='max',
        save_top_k=-1,
        save_last='link',
        save_on_exception=cfg.training.get('save_on_interruption', None),
        every_n_train_steps=cfg.training.save_per_steps,
        save_weights_only=cfg.training.save_weights_only,
    )

    # Logger selection based on config
    logger_type = cfg.training.get('logger', 'wandb').lower()
    if logger_type == 'wandb':
        # Use Weights & Biases logger
        logger = WandbLogger(
            project=cfg.training.wandb_project,
            name=run_name,
            id=cfg.training.get('wandb_resume_id', None),
            resume='allow',
        )
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if logger.experiment.config:
            # Merge with existing config, giving priority to existing values
            logger.experiment.config.update(cfg_dict, allow_val_change=True)
        else:
            # If no existing config, set it directly
            logger.experiment.config.update(cfg_dict)
    elif logger_type == 'tensorboard':
        # Use TensorBoard logger
        tensorboard_log_dir = Path('logs', run_name)
        logger = TensorBoardLogger(
            save_dir=tensorboard_log_dir,
            name=None,  # Use the directory as is without adding another subfolder
            version='',  # Don't add version subdirectory
        )
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")

    callbacks = [checkpoint_callback, CustomProgressBar(), EnsureFinalValidationCallback()]
    if lr_scheduler is not None:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed',
        accumulate_grad_batches=cfg.training.grad_accumulation_steps,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=cfg.training.test_per_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm='norm',
        log_every_n_steps=cfg.training.log_every_n_steps,
        # profiler=SimpleProfiler(dirpath='logs', filename='simple_profile'),
    )

    if hasattr(optimizer, 'train'):
        optimizer.train()

    # train_sampler = WeightedSampler(train_dataset.cache['weight'], replacement=True)
    train_sampler = WeightedRandomSampler(train_dataset.cache['weight'], len(train_dataset), replacement=True)
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            # shuffle=True,
            sampler=train_sampler,
            drop_last=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            persistent_workers=True,
            collate_fn=collate_fn,
        ),
        ckpt_path=resume_ckpt,
    )

if __name__ == "__main__":
    main()
