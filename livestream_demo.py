import os
import cv2
import torch
import multiprocessing
from argparse import ArgumentParser
from omegaconf import OmegaConf

# from chatdemo.server import ChatAvatarServer
from chatdemo.new_server import ChatAvatarServer

def parse_args():
    parser = ArgumentParser()

    group = parser.add_argument_group("Animation options")
    group.add_argument(
            "--cfg",
            type=str,
            required=True,
            help="model config file",
    )
    group.add_argument(
            "--server_host",
            type=str,
            required=False,
            default="0.0.0.0",
            help="ip address server listening to",
    )
    group.add_argument(
            "--server_port",
            type=int,
            required=False,
            default=12543,
            help="port server listening to",
    )
    group.add_argument(
            "--checkpoint",
            type=str,
            default='checkpoints/diffar_vocaset_241120.ckpt',
            required=False,
            help="path to model weights",
    )
    group.add_argument(
            "--id",
            type=str,
            default='FaceTalk_170731_00024_TA',
            required=False,
            help="animation identity",
    )
    group.add_argument(
            "--id_dim",
            type=int,
            default=8,
            required=False,
            help="animation dim",
    )
    group.add_argument(
            "--template",
            type=str,
            default='/nas/home/yangyifan/Code/3dFacialAnimation/DiffSpeaker/datasets/vocaset/templates.pkl',
            required=False,
            help="template file",
    )
    group.add_argument(
            "--ply",
            type=str,
            default='/nas/home/yangyifan/Code/3dFacialAnimation/DiffSpeaker/datasets/vocaset/templates/FLAME_sample.ply',
            required=False,
            help="ply file",
    )
    group.add_argument(
            "--device",
            type=str,
            default='cuda',
            required=False,
            help="accelerator",
    )

    params = parser.parse_args()

    return params

def main(args):
    cfg = OmegaConf.load(args.cfg)

    server = ChatAvatarServer(cfg=cfg, args=args)
    server.start_render()

if __name__ == "__main__":
#     torch.multiprocessing.set_start_method("spawn")

    args = parse_args()
    main(args)