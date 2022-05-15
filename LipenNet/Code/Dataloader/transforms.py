from PIL import Image, ImageStat, ImageOps, ImageEnhance
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
import os
import random
from Code.Profile.profileloader import Hparams

class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return T.functional.rotate(x, angle)


class AddGaussianNoise(object):  # this is dangerous do not use!!!
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class EnhanceBrightness(object):  # Karol's work
    '''This is custom transform class
        It creates bright circle in the center of image
        when initializing you can set
        brightness - float value determine minimal brightness would be product image, should be in <0, 1) <- dimmer <1 , max_bright) <- lighter
        probability - float value (0,1> determines the logically of preforming the transformation
        max_bright - float value (not smaller than bright) maximal brightness of the product image
        when you call it for specific picture it performs call method. Original design Karol Dziki, altered by ATM'''

    def __init__(self, bright: float = 2.5, max_bright: float = 3.0, probability: float = 1.0):
        if bright < 0: return 1
        if max_bright < bright: return 1
        self.max_bright: float = max_bright
        self.bright: float = bright
        self.probability: float = probability

    def __call__(self, img):
        fate = random.random()  # rand number from (0,1>
        fate_bright = random.random() * abs(self.max_bright - self.bright) + self.bright  # rand number from (0,1>
        if fate <= self.probability:
            return ImageEnhance.Brightness(img).enhance(fate_bright)
        else:
            return img  # do nothing


full_Transform = T.Compose([
    T.Resize(RESIZE_SIZE),  ## 244p

    T.RandomVerticalFlip(p=0.3),
    T.RandomHorizontalFlip(p=0.3),
    T.transforms.RandomApply(
        [T.transforms.ColorJitter(brightness=(0.9, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=(-0.5, 0.5))],
        p=0.5),
    T.transforms.ToTensor(),

    # T.Normalize(mean=[0.4784, 0.4712, 0.4662],
    #            std=[0.2442, 0.2469, 0.2409]),
    T.transforms.RandomApply(
        [AddGaussianNoise(0., 0.005)],
        p=0.41
    ),
    T.transforms.RandomApply(
        [T.transforms.GaussianBlur((1, 9), (0.1, 5))],
        p=0.3),

    T.transforms.ToPILImage(),
    EnhanceBrightness(bright=1.1, max_bright=1.6, probability=0.2),
    T.RandomInvert(p=0.1),
    T.RandomEqualize(p=0.3),
    T.RandomGrayscale(p=0.1),
    T.transforms.RandomApply(
        [T.RandomRotation(degrees=(0, 360))], p=0.5
    )])


class LipenTransform:
    transform = None

    def __init__(self, full_augmentation: bool, hparams: Hparams):

        mean = [0.4784, 0.4712, 0.4662]
        std = [0.2442, 0.2469, 0.2409]

        if full_augmentation:
            self.transform = T.Compose([
                T.Resize(resize_size),
                T.RandomVerticalFlip(p=0.3),
                T.RandomHorizontalFlip(p=0.3),
                T.transforms.RandomApply(
                    [T.transforms.ColorJitter(brightness=(0.9, 1), contrast=(0.5, 1), saturation=(0.5, 1),
                                              hue=(-0.5, 0.5))],
                    p=0.5),
                T.transforms.ToTensor(),

                T.Normalize(mean=[0.4784, 0.4712, 0.4662],
                            std=[0.2442, 0.2469, 0.2409]),
                T.transforms.RandomApply(
                    [AddGaussianNoise(0., 0.005)],
                    p=0.41
                ),
                T.transforms.RandomApply(
                    [T.transforms.GaussianBlur((1, 9), (0.1, 5))],
                    p=0.3),

                T.transforms.ToPILImage(),
                EnhanceBrightness(bright=1.1, max_bright=1.6, probability=0.2),
                T.RandomInvert(p=0.1),
                T.RandomEqualize(p=0.3),
                T.RandomGrayscale(p=0.1),
                T.transforms.RandomApply(
                    [T.RandomRotation(degrees=(0, 360))], p=0.5
                ),
            ])

        else:
            self.transform = T.Compose([
                T.Resize(resize_size),
                RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
            ])
