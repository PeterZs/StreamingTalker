import os
import importlib

from argparse import ArgumentParser
from omegaconf import OmegaConf

def parse_args(phase="train"):
    parser = ArgumentParser()

    group = parser.add_argument_group("Training options")
    if phase in ["train", "test", "demo"]:
        group.add_argument(
            "--exp",
            type=str,
            required=False,
            default="diffar",
            help="experiment name",
        )
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/config.yaml",
            help="config file",
        )
        group.add_argument(
            "--frame_rate",
            type=float,
            default=30,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--resume",
            type=str,
            required=False,
            help="resume from a checkpoint",
        )
        group.add_argument(
            "--batch_size",
            type=int,
            required=False,
            help="training batch size")
        group.add_argument(
            "--device",
            type=int,
            nargs="+",
            required=False)
        group.add_argument(
            "--dir",
            type=str,
            required=False,
            help="evaluate existing npys")
        group.add_argument(
            "--debug",
            action="store_true",
            required=False,
            help="debug mode"
        )
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            help="program mode"
        )

    if phase == "demo":
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--vertice",
            type=str,
            required=False,
            help="input vertices",
        )
        group.add_argument(
            "--output_dir",
            type=str,
            required=False,
            help="output dir",
        )
        group.add_argument(
            "--template",
            type=str,
            required=False,
            help="template path",
        )
        group.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="output seperate or combined npy file",
        )
        group.add_argument(
            "--id",
            type=str,
            required=True,
            help="the candiate subect identity",
        )
        group.add_argument(
            "--ply",
            type=str,
            required=True,
        )
        group.add_argument(
            "--split",
            type=str,
            required=False,
            help="train or val",
        )

    params = parser.parse_args()

    # merge config from files
    cfg_base = OmegaConf.load('./configs/base.yaml')
    cfg = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    cfg.NAME = params.exp if params.exp else cfg.NAME

    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEBUG = params.debug if params.debug is not None else cfg.DEBUG
        cfg.TRAIN.RESUME = params.resume if params.resume else cfg.TRAIN.RESUME

        if phase == "test":
            cfg.DEBUG = False
            # cfg.DEVICE = [0]

    if phase == "demo":
        cfg.DEMO.FRAME_RATE = params.frame_rate
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.VERTICE = params.vertice
        cfg.DEMO.CHECKPOINTS = params.checkpoint
        cfg.DEMO.TEMPLATE = params.template
        cfg.DEMO.ID = params.id
        cfg.DEMO.PLY = params.ply
        cfg.DEMO.FOLDER = params.output_dir
        cfg.DEMO.SPLIT = params.split
            
    # debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug_" + cfg.NAME
        cfg.LOGGER.WANDB.OFFLINE = True
        cfg.LOGGER.VAL_EPOCH = 1

    if params.mode is not None:
        cfg.MODE = params.mode
        cfg.NAME = cfg.MODE + cfg.NAME

    return cfg
