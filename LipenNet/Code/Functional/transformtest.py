from PIL import Image, ImageStat, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch

import random
import sys

import Code.Profile.profileloader as pl
import Code.Dataloader.lipenset as dl
import Code.Dataloader.transforms as tr
import Code.Protocol.enums as en


def plot(sour_img, imgs):

    num_rows = int ((len(imgs) + 1) / 5) + ((len(imgs) + 1) % 2)
    num_cols = 2
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
            axs[row_idx, col_idx].imshow(imgs[img_idx].squeeze().permute(1,2,0))
            axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[] )
            img_idx += 1

    plt.tight_layout()
    #plt.figure(figsize=(20, 20))
    plt.show()


def main():
    hparams: pl.Hparams = pl.loadProfile(sys.argv)
    train_loader, _,_ = dl.loadData(hparams)
    train_loader = [next(iter(train_loader))]

    seed = int(random.random() * 100 * random.random()) if True else 1525
    torch.manual_seed(seed)

    transform = tr.LipenTransform(augmentation_type=en.AugmentationType.Online, hparams=hparams)

    for data in train_loader:
        images = data["path"]
        for image in images:
            pimage = Image.open(image)
            pimage = ImageOps.exif_transpose(pimage)
            sub_imgs = [transform.transform(pimage) for _ in range(4)]
            plot(pimage, sub_imgs)
            pimage.close()







if __name__ == '__main__':
    main()