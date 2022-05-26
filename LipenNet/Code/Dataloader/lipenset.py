import os
import random
import math
import numpy as np
import pickle
import sys

import skimage
from torch.utils.data import Dataset, DataLoader


import Code.Protocol.enums as en
from Code.Profile.profileloader import Hparams
from skimage import io
from PIL import Image

import Code.Dataloader.transforms as t
import Code.Dataloader.datatools as dt


class Lipenset(Dataset):
    def __init__(self, hparams : Hparams, dataset_type:en.DatasetType, shuffle=False):
        self.dataset_path = hparams['data_dir'] + "/" + hparams['dataset_dir']
        self.dataset_type = dataset_type
        self.shuffle = shuffle
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

        self.transform_tool = None
        match (self.augmentation_type or dataset_type):
            case en.AugmentationType.Without:
                self.transform_tool = t.LipenTransform(augmentation_type=en.AugmentationType.Without, hparams=hparams)
            case en.DatasetType.Testset | en.DatasetType.ValSet | en.AugmentationType.Normalize:
                self.transform_tool = t.LipenTransform(augmentation_type=en.AugmentationType.Normalize, hparams=hparams)
            case en.AugmentationType.Rotation:
                self.transform_tool = t.LipenTransform(augmentation_type=en.AugmentationType.Rotation, hparams=hparams)
            case en.AugmentationType.Online:
                self.transform_tool = t.LipenTransform(augmentation_type=en.AugmentationType.Online, hparams=hparams)
            case en.AugmentationType.Offline:
                self.transform_tool = t.LipenTransform(augmentation_type=en.AugmentationType.Offline, hparams=hparams)

        self.images :list[dict] = []
        image_files = dt.getImageFiles(self.dataset_path) #TODO - TypeError: an integer is required - warning
        self.image_amount = len(image_files)
        if self.image_amount == 0:
            print("No Images were founded")
            sys.exit(1)

        for image_file in image_files:
            with open(self.label_filepath, "r", encoding='utf-8') as label_file:
                for label_line in label_file:
                    name = label_line.split(";")[0]
                    if name == "\n": continue
                    if name == "Name": continue
                    if name != image_file.split("/")[-2]+"/"+image_file.split("/")[-1]: continue
                    label = int(label_line.split(";")[1])
                    sub_label = int(label_line.split(";")[3])
                    extra_label = int(label_line.split(";")[3])
                    image_dict = {"label":label,"path":image_file,"sub":sub_label,"extra":extra_label}
                    self.images.append(image_dict)
        if len(self.images) != self.image_amount:
            print("Different image number in csv and dirs")
            sys.exit(1)


        # Mix up the data
        if self.shuffle:
            random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_dict = self.images[idx]
        imagep = Image.open(image_dict["path"])
        image = self.transform_tool.transform(imagep)
        image_dict["image"] = image
        imagep.close()
        #Add weights
        return image_dict



def loadData(hparams : Hparams, load_train:bool = False,load_val:bool= False,load_test:bool= False):
    trainset = None
    valset = None
    testset = None

    if load_train:
        trainset = Lipenset(hparams,en.DatasetType.Trainset,shuffle=True)
    if load_val:
        valset = Lipenset(hparams, en.DatasetType.ValSet, shuffle=False)
    if load_test:
        testset = Lipenset(hparams, en.DatasetType.Testset, shuffle=False)

    train_loader = None
    eval_loader = None
    test_loader = None

    if load_train:
        train_loader = DataLoader(trainset, batch_size=hparams['train_batch_size'], shuffle=True)
    if load_val:
        eval_loader = DataLoader(valset, batch_size=hparams['val_batch_size'], shuffle=False)
    if load_test:
        test_loader = DataLoader(testset, batch_size=hparams['test_batch_size'], shuffle=False)

    return train_loader, eval_loader, test_loader
