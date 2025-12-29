import os
import pickle
import torch
import time
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from configs.config import parse_args
from algorithms.models import get_model
from utils.logger_utils import create_logger
from utils.demo_utils import animate, load_audio

train_subjects_voca = {
        'FaceTalk_170728_03272_TA': 0,
        'FaceTalk_170904_00128_TA': 1,
        'FaceTalk_170725_00137_TA': 2,
        'FaceTalk_170915_00223_TA': 3,
        'FaceTalk_170811_03274_TA': 4,
        'FaceTalk_170913_03279_TA': 5,
        'FaceTalk_170904_03276_TA': 6,
        'FaceTalk_170912_03278_TA': 7,
    }

train_subjects_biwi = {
        "F2": 0,
        "F3": 1,
        "F4": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
    }

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
    model = get_model(cfg=cfg)
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]
    
     # this is not required, since the sequence has flexible length
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # load audio input
    if cfg.DEMO.EXAMPLE: 
        logger.info("Loading audio from {}".format(cfg.DEMO.EXAMPLE))
        wav_path = cfg.DEMO.EXAMPLE
        assert os.path.exists(wav_path), 'audio does not exist'
        audio = load_audio(wav_path)
    else:
        raise NotImplemented

    # load the template
    logger.info("Loading template mesh from {}".format(cfg.DEMO.TEMPLATE))
    template_file = cfg.DEMO.TEMPLATE
    with open(template_file, 'rb') as fin:
        template = pickle.load(fin,encoding='latin1')
        subject_id = cfg.DEMO.ID
        assert subject_id in template, f'{subject_id} is not a subject included'
        template = torch.Tensor(template[subject_id].reshape(-1))

    if cfg.DATASET.NAME == 'vocaset':
        train_subjects = train_subjects_voca
    elif cfg.DATASET.NAME == 'biwi':
        train_subjects = train_subjects_biwi

    # load the subject id
    if subject_id in train_subjects:
        id_idx = train_subjects[subject_id]
        id = torch.zeros((1,cfg.MODEL.DIFF_AR_DEC.id_dim))
        id[0, id_idx] = 1
    else:
        id = torch.zeros((1,cfg.MODEL.DIFF_AR_DEC.id_dim))
        id[0, 0] = 1

    input_data = {
        'id': id.to(device),
        'audio': audio.to(device),
        'template': template.to(device),
        'vertice': None
    }
    # visulize trainset performance
    if cfg.DEMO.SPLIT == 'train':
        vertice_path = cfg.DEMO.VERTICE
        if not os.path.exists(vertice_path):
            print("vertice file not found")
            return None
        else:
            print(f"Loading vertices from{vertice_path}")
            # input_data["vertice"] = torch.FloatTensor(np.load(vertice_path,allow_pickle=True)[::2,:]).to(device)
            input_data["vertice"] = torch.FloatTensor(np.load(vertice_path,allow_pickle=True)).to(device)
            vertices = input_data["vertice"]

    logger.info("Start making predictions")

    with torch.no_grad():
        # output_file = "time.txt"
        # t1 = time.time()
        
        vertices_pred = model.predict(input_data)
        vertices_pred = vertices_pred.squeeze().cpu().numpy()

        # t2 = time.time()
        # with open(output_file, 'a') as f:
        #     f.write(test_name + " " + str(t2-t1) + "\n")
            
    # borrowed from faceformer    
    output_dir = os.path.join(cfg.FOLDER, str(cfg.MODEL.TYPE), str(cfg.NAME), "samples_" + cfg.TIME)
    pred_file_name = os.path.join(output_dir, 'vertice_pred' + '.mp4')
    gt_file_name = os.path.join(output_dir, 'vertice_gt' + '.mp4')
    output_file = os.path.join(output_dir,'output.mp4')
    
    animate(vertices_pred, wav_path, pred_file_name, cfg.DEMO.PLY, fps=cfg.DEMO.FPS, use_tqdm=True, multi_process=True)

    if cfg.DEMO.SPLIT == 'train':
        vertices = vertices.cpu().numpy()
        animate(vertices, wav_path, gt_file_name, cfg.DEMO.PLY, fps=cfg.DEMO.FPS, use_tqdm=True, multi_process=True)
        cmd = " ".join(['ffmpeg', '-i', gt_file_name , '-i', pred_file_name , '-filter_complex "[0:v][1:v]hstack" -c:a copy', output_file, ])
        os.system(cmd)

if __name__ == "__main__":
    main()