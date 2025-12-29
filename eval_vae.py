import os
import pickle
import torch
import time
import numpy as np
import torch.nn.functional as F

from configs.config import parse_args
from algorithms.models.vae_trainer import VAETrainer
from utils.logger_utils import create_logger
from utils.demo_utils import animate


def main():
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER

    if cfg.ACCELERATOR == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create logger
    dataset = cfg.DATASET.NAME
    logger = create_logger(cfg, phase="demo")

    # create model & load ckpt
    model = VAETrainer(cfg)
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # load the template
    logger.info("Loading template mesh from {}".format(cfg.DEMO.TEMPLATE))
    template_file = cfg.DEMO.TEMPLATE
    with open(template_file, 'rb') as fin:
        template = pickle.load(fin,encoding='latin1')
        subject_id = cfg.DEMO.ID
        assert subject_id in template, f'{subject_id} is not a subject included'
        template = torch.Tensor(template[subject_id].reshape(-1))

    # load vertices
    vertice_path = cfg.DEMO.VERTICE
    if cfg.DATASET.NAME == 'biwi':
        vertices = torch.FloatTensor(np.load(vertice_path,allow_pickle=True)).to(device)
    elif cfg.DATASET.NAME == 'vocaset':
        vertices = torch.FloatTensor(np.load(vertice_path,allow_pickle=True)[::2,:]).to(device)

    logger.info("Start making predictions")
    
    with torch.no_grad():
        vertices_pred, quant_loss, info = model.autoencoder(vertices[None,...], template[None,...].to(device))
        vertices_pred = vertices_pred.squeeze().cpu().numpy()
        vertices = vertices.cpu().numpy()
            
    output_dir = os.path.join(cfg.FOLDER, str(cfg.MODEL.TYPE), str(cfg.NAME), "samples_" + cfg.TIME)
    pred_file_name = os.path.join(output_dir, 'vertice_pred' + '.mp4')
    gt_file_name = os.path.join(output_dir, 'vertice_gt' + '.mp4')
    output_file = os.path.join(output_dir,'output.mp4')

    wav_path = None
    
    animate(vertices_pred, wav_path, pred_file_name, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)
    animate(vertices, wav_path, gt_file_name, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)

    cmd = " ".join(['ffmpeg', '-i', gt_file_name , '-i', pred_file_name , '-filter_complex "[0:v][1:v]hstack" -c:a copy', output_file, ])
    os.system(cmd)

if __name__ == "__main__":
    main()