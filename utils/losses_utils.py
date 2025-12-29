import os
import torch
import numpy as np
import pickle

with open(os.path.join("data/biwi/regions", "lve.txt")) as f:
    maps = f.read().split(", ")
    mouth_map = [int(i) for i in maps]

with open(os.path.join("data/biwi/regions", "fdd.txt")) as f:
    maps = f.read().split(", ")
    upper_map = [int(i) for i in maps]

with open(os.path.join("data/vocaset", "FLAME_masks.pkl"), "rb") as f:
    masks = pickle.load(f, encoding='latin1')
    vocaset_mouth_map = masks["lips"].tolist()
    vocaset_upper_map = masks["forehead"].tolist() + masks["eye_region"].tolist()
    vocaset_upper_map = list(set(vocaset_upper_map))


def vocaset_upper_face_variance(motion, ):
    L2_dis_upper = np.array([np.square(motion[:,v, :]) for v in vocaset_upper_map])
    L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
    L2_dis_upper = np.sum(L2_dis_upper,axis=2)
    L2_dis_upper = np.std(L2_dis_upper, axis=0)
    motion_std = np.mean(L2_dis_upper)
    return torch.tensor(motion_std).float() #torch.from_numpy(motion_std).float()

def vocaset_mouth_distance(vertices_gt, vertices_pred):
    L2_dis = np.array([np.square(vertices_gt[:,v, :] - vertices_pred[:,v, :]) for v in vocaset_mouth_map])
    L2_dis = np.transpose(L2_dis, (1,0,2)) # (V, N, 3) -> (N, V, 3)
    L2_dis = np.sum(L2_dis, axis=2) # (N, V, 3) -> (N, V)
    L2_dis = np.max(L2_dis, axis=1) # (N, V) -> (N)
    return torch.tensor(L2_dis).float()

def biwi_upper_face_variance(motion, ):
    L2_dis_upper = np.array([np.square(motion[:,v, :]) for v in upper_map])
    L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
    L2_dis_upper = np.sum(L2_dis_upper,axis=2)
    L2_dis_upper = np.std(L2_dis_upper, axis=0)
    motion_std = np.mean(L2_dis_upper)
    return torch.tensor(motion_std).float() #torch.from_numpy(motion_std).float()

def biwi_mouth_distance(vertices_gt, vertices_pred):
    L2_dis = np.array([np.square(vertices_gt[:,v, :] - vertices_pred[:,v, :]) for v in mouth_map])
    L2_dis = np.transpose(L2_dis, (1,0,2)) # (V, N, 3) -> (N, V, 3)
    L2_dis = np.sum(L2_dis, axis=2) # (N, V, 3) -> (N, V)
    L2_dis = np.max(L2_dis, axis=1) # (N, V) -> (N)
    return torch.tensor(L2_dis).float()