import os
import torch
import pickle
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor
from collections import defaultdict

from utils.data_utils import load_data

class VOCASETDataset(Dataset):
    def __init__(self, 
                data, 
                subjects_dict, 
                data_type="train",
                ):

        self.data = data
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        # only train subjects have identity label
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        # seq_len(fps), latent_dim
        file_name = self.data[index]["name"]
        file_path = self.data[index]["path"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        elif self.data_type == "val":
            one_hot = self.one_hot_labels
        elif self.data_type == "test":
            subject = "_".join(file_name.split("_")[:-1])
            if subject in self.subjects_dict["train"]:
                one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            else:
                one_hot = self.one_hot_labels

        return {
            'audio':torch.FloatTensor(audio),
            'vertice':torch.FloatTensor(vertice), 
            'template':torch.FloatTensor(template), 
            'id':torch.FloatTensor(one_hot), 
            'file_name':file_name,
            'file_path':file_path,
        }

    def __len__(self):
        return len(self.data)

# pack dataloaders into pl DataModule
class VOCASETDataModule(pl.LightningDataModule):
    def __init__(self,
                cfg,
                phase,
                batch_size,
                num_workers,
                cpu_counts = 4,
                multiprocessing = False,
                collate_fn = None,
                prefetch_factor = 2,
                pin_memory = True,
                persistent_workers = True,
                **kwargs):
        super().__init__()

        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "prefetch_factor": prefetch_factor,
            "pin_memory": pin_memory, 
            "persistent_workers": persistent_workers
        }
    
        self.name = 'VOCASET'
        self.cfg = cfg
        self.phase = phase

        # select train,val,test identity
        self.subjects = {
            'train': [
                'FaceTalk_170728_03272_TA',
                'FaceTalk_170904_00128_TA',
                'FaceTalk_170725_00137_TA',
                'FaceTalk_170915_00223_TA',
                'FaceTalk_170811_03274_TA',
                'FaceTalk_170913_03279_TA',
                'FaceTalk_170904_03276_TA',
                'FaceTalk_170912_03278_TA'
            ],
            'val': [
                'FaceTalk_170811_03275_TA',
                'FaceTalk_170908_03277_TA'
            ],
            'test': [
                'FaceTalk_170809_00138_TA',
                'FaceTalk_170731_00024_TA'
            ]
        }

        # audio processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.root_dir = cfg.DATASET.DATA_ROOT
        self.audio_dir = 'wav'
        self.vertice_dir = 'vertices_npy'
        self.template_file = 'templates.pkl'

        self.splits = {
            "train": [],
            "val": [],
            "test": []
        }

        self.split_ranges = {
            "train": range(1,41),
            "val": range(21,41),
            "test": range(21,41)
        }

        # load dataset
        data = defaultdict(dict)
        with open(os.path.join(self.root_dir, self.template_file), 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        args_list = []
        for root, dirs, files in os.walk(os.path.join(self.root_dir, self.audio_dir)):
            for file in files:
                subject_id = "_".join(file.split("_")[:-1])
                sentence_id = int(file.split(".")[0][-2:])

                # eval-phase, only load the subjects in the test split
                if self.phase == "test":
                    if subject_id not in self.subjects["test"]:  
                        continue
                    if sentence_id not in self.split_ranges["test"]:
                        continue
                args_list.append((file, self.root_dir, processor, templates, self.audio_dir, self.vertice_dir, self.name, ))

        if multiprocessing:
            with Pool(processes = cpu_counts) as pool:
                results = pool.map(load_data, args_list)
                for result in results:
                    if result is not None:
                        k, v = result
                        data[k] = v
                    else:
                        print("data not loaded...")
        else:
            for args in tqdm(args_list):
                result = load_data(args)
                if result is not None:
                    k, v = result
                    data[k] = v
                else:
                    print("data not loaded...")

        for k, v in data.items():
            subject_id = "_".join(k.split("_")[:-1])
            sentence_id = int(k.split(".")[0][-2:])
            for split in ["train", "val", "test"]:
                if subject_id in self.subjects[split] and sentence_id in self.split_ranges[split]:
                    self.splits[split].append(v)

    def setup(self, stage):
        self.train = VOCASETDataset(
            data=self.splits["train"],
            subjects_dict=self.subjects,
            data_type="train"
        )
        self.val = VOCASETDataset(
            data=self.splits["val"],
            subjects_dict=self.subjects,
            data_type="val"
        )
        self.test = VOCASETDataset(
            data=self.splits["test"],
            subjects_dict=self.subjects,
            data_type="test"
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs.copy()
        dataloader_kwargs["batch_size"] = self.cfg.VAL.BATCH_SIZE
        dataloader_kwargs["num_workers"] = self.cfg.VAL.NUM_WORKERS
        dataloader_kwargs["shuffle"] = False
        return DataLoader(
            self.val,
            **dataloader_kwargs,
        )

    def test_dataloader(self):
        dataloader_kwargs = self.dataloader_kwargs.copy()
        dataloader_kwargs["batch_size"] = self.cfg.TEST.BATCH_SIZE
        dataloader_kwargs["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_kwargs["shuffle"] = False
        return DataLoader(
            self.test,
            **dataloader_kwargs,
        )
