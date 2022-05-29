import re
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import Code.Protocol.enums as en
from Code.Profile.profileloader import Hparams
from PIL import Image
import Code.Dataloader.transforms as t
import Code.Dataloader.datatools as dt
import numpy as np


class Lipenset(Dataset):
    def __init__(self, hparams : Hparams, dataset_type:en.DatasetType, shuffle=False):
        self.dataset_path = hparams['data_dir'] + "/" + hparams['dataset_dir']
        self.dataset_type = dataset_type
        self.shuffle = shuffle

        set_dir = ""
        match self.dataset_type:
            case en.DatasetType.Testset:
                set_dir = hparams['testset_dir']
            case en.DatasetType.ValSet:
                set_dir = hparams['valset_dir']
            case en.DatasetType.Trainset:
                set_dir = hparams['trainset_dir']
        self.dataset_path = self.dataset_path + "/" + set_dir
        self.dataset_name = hparams['dataset_name']
        self.label_filepath = hparams['data_dir'] + "/" + hparams['dataset_dir'] + "/" + hparams['label_filename']

        self.augmentation_type = hparams['augmentation_type']

        self.transform_tool = None
        if dataset_type in [en.DatasetType.Testset, en.DatasetType.ValSet]:
            self.transform_tool = t.LipenTransform(augmentation_type=en.AugmentationType.Normalize, hparams=hparams)
        else:
            self.transform_tool = t.LipenTransform(augmentation_type=self.augmentation_type, hparams=hparams)

        self.images = []
        image_files = dt.getImageFiles(self.dataset_path) #TODO - TypeError: an integer is required - warning
        self.image_amount = len(image_files)
        if self.image_amount == 0:
            print("No Images were founded")
            sys.exit(1)

        with open(self.label_filepath, "r", encoding='utf-8') as label_file:
            label_lines = label_file.readlines()[1:]
        label_lines = list(filter(lambda x: x != "\n", label_lines))

        image_file_pattern = re.compile(f".*{set_dir}/(.*)")
        image_file_parsed = [image_file_pattern.match(image_file).group(1) for image_file in image_files]
        label_lines_split_raw = [label_line.split(';') for label_line in label_lines]
        label_lines_split = list(filter(lambda x: x[0] in image_file_parsed, label_lines_split_raw))

        if len(label_lines_split) != len(image_files) or len(image_files) != self.image_amount:
            print("Different image number in csv and dirs")
            sys.exit(1)

        for label_line_info, image_file in zip(sorted(label_lines_split), sorted(image_files)):
            extras_code = int(label_line_info[3])
            is_hard = ((16 & extras_code) >> 4)
            weight = (2 * is_hard) | 1 - is_hard
            image_info = [image_file, int(label_line_info[1]), weight]
            image_info = np.array(image_info, dtype=object)
            self.images.append(image_info)

        self.images = np.array(self.images)

        # Mix up the data
        if self.shuffle:
            self.shuffle_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        imagep = Image.open(image_info[0])
        image = self.transform_tool.transform(imagep)
        imagep.close()
        image_info = image_info[1:]
        image_info = np.append(image_info, image.size()).astype("float32")
        image_info = np.append(image_info, image)
        return image_info

    def shuffle_images(self):
        np.random.shuffle(self.images)


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
        train_loader = DataLoader(trainset, batch_size=hparams['train_batch_size'], shuffle=False)
    if load_val:
        eval_loader = DataLoader(valset, batch_size=hparams['val_batch_size'], shuffle=False)
    if load_test:
        test_loader = DataLoader(testset, batch_size=hparams['test_batch_size'], shuffle=False)

    return train_loader, eval_loader, test_loader
