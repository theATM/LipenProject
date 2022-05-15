import os
import random
import mne
import math
import numpy as np
import pickle
import sys
from torch.utils.data import Dataset, DataLoader


import Code.Protocol.enums as en
import Code.Dataloader.datatools as dt
from Code.Profile.profileloader import Hparams
from PIL import Image

import Code.Dataloader.transforms as t
import Code.Dataloader.datatools as dt

class Lipenset(Dataset):
    def __init__(self, hparams : Hparams, dataset_type:en.DatasetType):
        self.dataset_path = hparams['data_dir'] + "/" + hparams['dataset_dir']
        self.dataset_type = dataset_type
        match self.dataset_type:
            case en.DatasetType.Testset:
                self.dataset_path = self.dataset_path + "/" + hparams['testset_dir']
            case en.DatasetType.ValSet:
                self.dataset_path = self.dataset_path + "/" + hparams['valset_dir']
            case en.DatasetType.Trainset:
                self.dataset_path = self.dataset_path + "/" + hparams['trainset_dir']
        self.dataset_name = hparams['dataset_name']
        self.label_filepath = hparams['data_dir'] + "/" + hparams['dataset_dir'] + "/" + hparams['label_filename']

        self.augmentation_type = hparams['augmentation_type']

        match self.augmentation_type and dataset_type:
            case en.DatasetType.Testset | en.DatasetType.ValSet | en.AugmentationType.Without:
                self.transform_tool = None
            case en.AugmentationType.Rotation:
                self.transform_tool = t.LipenTransform(full_augmentation=False, hparams=hparams)
            case en.AugmentationType.Online:
                self.transform_tool = t.LipenTransform(full_augmentation=True, hparams=hparams)

        self.images :list[dict] = []
        image_files = dt.getImageFiles(self.dataset_path,self.dataset_path)
        image_amount = len(image_files)
        if image_amount == 0:
            print("No Images were founded")
            sys.exit(1)

        for image_file in image_files:
            with open(self.label_filepath, "r", encoding='utf-8') as label_file:
                for label_line in label_file:
                    name = label_line.split(";")[0]
                    if name == "\n": continue
                    if name == "Name": continue
                    if name != image_file.split("/")[-1]: continue
                    label = int(label_line.split(";")[1])
                    image_dict = {"label":label,"path":image_file}
                    self.images.append(image_dict)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_dict = self.images[idx]
        with Image.open(image_dict["path"]) as image:
            if self.transform_tool:
                image = self.transform_tool.transform(image)
            image_dict["image"] = image
            return image_dict



def loadData(hparams):
