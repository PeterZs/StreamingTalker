from .vocaset import VOCASETDataModule
from .biwi import BIWIDataModule

from utils.data_utils import collate_fn

def get_datasets(cfg, phase='train'):
    dataset_name = eval(f"cfg.DATASET.NAME")
    if dataset_name.lower() in ['vocaset']:
        dataset = VOCASETDataModule(
            cfg = cfg,
            phase = phase,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=collate_fn,
        )
    if dataset_name.lower() in ['biwi']:
        dataset = BIWIDataModule(
            cfg = cfg,
            phase = phase,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=collate_fn,
        )

    return dataset