import os
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf
from collections import OrderedDict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import get_datasets
from utils.logger_utils import create_logger
from utils.progress import ProgressLogger
from configs.config import parse_args
from algorithms.models import get_model

def train():
    cfg = parse_args()

    # create logger
    logger = create_logger(cfg, phase="train")

    # resume
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        backcfg = cfg.TRAIN.copy()
        losscfg = cfg.LOSS.copy()
        if os.path.exists(resume):
            file_list = sorted(os.listdir(resume), reverse=True)
            for item in file_list:
                if item.endswith(".yaml"):
                    cfg = OmegaConf.load(os.path.join(resume, item))
                    cfg.TRAIN = backcfg
                    cfg.LOSS = losscfg
                    break
            checkpoints = sorted(os.listdir(os.path.join(
                resume, "checkpoints")),
                                 key=lambda x: int(x[6:-5]),
                                 reverse=True)
            for checkpoint in checkpoints:
                if "epoch=" in checkpoint:
                    cfg.TRAIN.PRETRAINED = os.path.join(
                        resume, "checkpoints", checkpoint)
                    break
            if os.path.exists(os.path.join(resume, "wandb")):
                wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")),
                                    reverse=True)
                for item in wandb_list:
                    if "run-" in item:
                        cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]
            logger.info("Resume from {}".format(resume))
        else:
            raise ValueError("Resume path is not right.")
        
    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # tensorboard logger and wandb logger
    loggers = []
    if cfg.LOGGER.WANDB.PROJECT:
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.OFFLINE,
            id=cfg.LOGGER.WANDB.RESUME_ID,
            save_dir=cfg.FOLDER_EXP,
            version="",
            name=cfg.NAME,
            anonymous=False,
            log_model=False,
        )
        loggers.append(wandb_logger)
    if cfg.LOGGER.TENSORBOARD:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.FOLDER_EXP,
                                                 sub_dir="tensorboard",
                                                 version="",
                                                 name="")
        loggers.append(tb_logger)
    # logger.info(OmegaConf.to_yaml(cfg))

    # create datamodule
    datamodule = get_datasets(cfg=cfg)
    logger.info("{} Datamodule initialized".format("".join(
        cfg.DATASET.NAME)))

    # create model
    model = get_model(cfg=cfg)
    logger.info("{} Model initialized".format(cfg.MODEL.TYPE))

    metric_monitor = {
        "Diff_loss_Train": "diff_loss/train",
        "Diff_loss_Val":"diff_loss/val",
        "Train_vertice_recon": "vertice/enc/train",
        "Train_vertice_reconv": "vertice/encv/train",
        "Train_lip_recon": "lip/enc/train",
        "Train_lip_reconv": "lip/encv/train",
        "Val_vertice_recon": "vertice/enc/val",
        "Val_vertice_reconv": "vertice/encv/val",
        "Val_lip_recon": "lip/enc/val",
        "Val_lip_reconv": "lip/encv/val",
    }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch:02d}",
            monitor="step",
            mode="max",
            every_n_epochs=cfg.LOGGER.SAVE_CKPT_EPOCH,
            save_top_k=-1,
            save_last=False,
            save_on_train_epoch_end=True,
        ),
    ]
    logger.info("Callbacks initialized")

    if len(cfg.DEVICE) > 1:
        ddp_strategy = "ddp"
        logger.info("Trained with DDP")
    else:
        ddp_strategy = None

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy=ddp_strategy,
        default_root_dir=cfg.FOLDER_EXP,
        log_every_n_steps=cfg.LOGGER.LOG_EPOCH,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EPOCH,
        num_sanity_val_steps=0,
    )
    logger.info("Trainer initialized")

    if cfg.TRAIN.PRETRAINED:
        logger.info("resume from {}".format(
            cfg.TRAIN.PRETRAINED))
        logger.info("Attention! VAE will be recovered")
        state_dict = torch.load(cfg.TRAIN.PRETRAINED,
                                map_location="cpu")["state_dict"]

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    # fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule = datamodule,
                    ckpt_path = cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule = datamodule)

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")

if __name__=="__main__":
    train()