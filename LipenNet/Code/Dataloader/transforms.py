from PIL import Image, ImageStat, ImageOps, ImageEnhance
import pickle
import torch
import torchvision.transforms as T
import Code.Protocol.enums as en
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
    """This is custom transform class
        It creates bright circle in the center of image
        when initializing you can set
        brightness - float value determine minimal brightness would be product image, should be in <0, 1) <- dimmer <1 , max_bright) <- lighter
        probability - float value (0,1> determines the logically of preforming the transformation
        max_bright - float value (not smaller than bright) maximal brightness of the product image
        when you call it for specific picture it performs call method. Original design Karol Dziki, altered by ATM"""

    def __init__(self, bright: float = 2.5, max_bright: float = 3.0, probability: float = 1.0):
        if bright < 0 or max_bright < bright:
            return  # error?
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


class LipenTransform:
    transform = None

    def __init__(self, augmentation_type: en.AugmentationType, hparams: Hparams):
        with open(f"{hparams['data_dir']}/{hparams['dataset_dir']}/{hparams['normalization_filename']}", 'rb') as n_f:
            mean, std = pickle.load(n_f)

        if augmentation_type in (en.AugmentationType.Online, en.AugmentationType.Offline):
            self.transform = T.Compose([
                T.Resize(hparams['resize_size']),
                T.RandomVerticalFlip(hparams['vertical_flip_prob']),
                T.RandomHorizontalFlip(hparams['horizontal_flip_prob']),
                RandomRotationTransform(hparams['rotate_angles']),
                T.transforms.RandomApply(
                    [T.RandomRotation(degrees=hparams['random_rotation_degrees'])],
                    p=hparams['random_rotation_prob']),

                T.transforms.RandomApply(
                    [T.transforms.ColorJitter(hparams['color_jitter_brightness'],
                                              hparams['color_jitter_contrast'],
                                              hparams['color_jitter_saturation'],
                                              hparams['color_jitter_hue'])],
                    p=hparams['color_jitter_prob']),

                T.RandomEqualize(hparams['random_equalize_prob']),
                T.RandomInvert(hparams['random_invert_prob']),
                EnhanceBrightness(hparams['enhance_brightness_brightness_intensity'],
                                  hparams['enhance_brightness_max_brightness'],
                                  hparams['enhance_brightness_prob']),
                T.RandomGrayscale(hparams['random_greyscale_prob']),

                T.ToTensor(),

                T.transforms.RandomApply(
                    [AddGaussianNoise(hparams['gaussian_noise_mean'], hparams['gaussian_noise_std'])],
                    p=hparams['gaussian_noise_prob']),

                T.transforms.RandomApply(
                    [T.transforms.GaussianBlur(hparams['gaussian_blur_kernel_size'], hparams['gaussian_blur_sigma'])],
                    p=hparams['gaussian_blur_prob'])
            ])
            if augmentation_type == en.AugmentationType.Online:
                self.transform = T.Compose([self.transform, T.Normalize(mean, std)])
        elif augmentation_type == en.AugmentationType.Rotation:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(hparams['resize_size']),
                RandomRotationTransform(hparams['rotate_angles']),
            ])
        elif augmentation_type == en.AugmentationType.Normalize:
            self.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=mean,std=std)
                ])
        elif augmentation_type == en.AugmentationType.Without:
            self.transform = T.Compose([
                    T.ToTensor(),
                    T.Resize(hparams['resize_size']),
                ])