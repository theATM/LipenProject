from PIL import Image, ImageStat, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import random
import os
import utilities

IN_IMAGES_PATH = "inpictures/" #Must be with '/' at the end
OUT_IMAGES_PATH = "outpictures/" #Must be with '/' at the end
RESIZE_SIZE = (244,244)


torch.manual_seed(1525)


class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return T.functional.rotate(x, angle)


class AddGaussianNoise(object): #this is dangerous do not use!!!
    def __init__(self, mean=0., std=0.4):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())  * 0.1 #* self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


my_Transform = T.Compose([
    #T.Resize(RESIZE_SIZE),
    #T.RandomVerticalFlip(p=0.3),
    #T.RandomInvert(p=0.1),
    T.RandomHorizontalFlip(p=0.3),

    T.transforms.ToTensor(),
    #T.Normalize(),
    #T.RandomEqualize(p=1),
    T.transforms.RandomApply(
            [AddGaussianNoise(0.1, 0.1)],
            p=1
        ),
    #T.transforms.GaussianBlur((5,9),(1,5)),
    T.transforms.ToPILImage(),

    #T.RandomGrayscale(p=0.1),
    #T.transforms.RandomApply(
    #    [T.RandomRotation(degrees=(0, 360))],p=0.5
    #)


])

def plot(sour_img, imgs):

    num_rows = int ((len(imgs) + 1) / 5) + ((len(imgs) + 1) % 5)
    num_cols = 5
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)


    axs[0, 0].set(title='Original image')
    axs[0, 0].title.set_size(8)
    axs[0,0].imshow(sour_img)
    axs[0,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    img_idx = 0
    for row_idx in range(num_rows):
        for col_idx in range (num_cols):
            if row_idx == 0 and col_idx == 0 : continue
            if img_idx >= len(imgs) :
                ax = axs[row_idx, col_idx]
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], frame_on=False)
                continue
            axs[row_idx, col_idx].imshow( imgs[img_idx] )
            axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[] )
            img_idx += 1

    plt.tight_layout()
    plt.show()


def main():
    print("Witaj w programie augmentującym otagowane już zdjęcia")
    # Check imput
    if not os.path.isdir(IN_IMAGES_PATH):
        print("Nie znaleziono filderu ze zdjęciami")
        print("Utwórz folder \"" + IN_IMAGES_PATH[:-1] + "\"")
        return 1
    if IN_IMAGES_PATH[-1] != '/' or OUT_IMAGES_PATH[-1] != '/':
        print("Dodaj znaki \/ na koniec nazw folderów ze zdjęciami")
        return 1
    # Create out dir
    if not os.path.isdir(OUT_IMAGES_PATH):
        os.mkdir(str(OUT_IMAGES_PATH))

    image_list = utilities.getFiles(IN_IMAGES_PATH, IN_IMAGES_PATH, OUT_IMAGES_PATH, True)
    image_amount = len(image_list)
    if image_amount == 0:
        print("Nie znaleziono żadnych zdjęć ")
        return 1

    # Perform Computations:
    for image in image_list:
        ti_m = os.path.getmtime(IN_IMAGES_PATH + image)
        # Open
        pimage = Image.open(IN_IMAGES_PATH + image)
        pimage = ImageOps.exif_transpose(pimage)
        sub_imgs = [my_Transform(pimage) for _ in range(4)]
        sub_img = my_Transform(pimage)
        plot(pimage, sub_imgs)
        '''
        # Rotate
        resimg = my_Transform(pimage)
        # Compress
        if pimage.size[0] > RESIZE_SIZE[0] and pimage.size[1] > RESIZE_SIZE[1]:
            resimg = resimg.resize(RESIZE_SIZE)
        '''
        # Save
        sub_img.save(OUT_IMAGES_PATH + image)
        # Set old modification time
        os.utime(OUT_IMAGES_PATH + image, (ti_m, ti_m))

    print("Koniec")



if __name__ == '__main__':
    main()