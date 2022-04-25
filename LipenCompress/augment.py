from PIL import Image, ImageStat, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import random
import os
import utilities

IN_IMAGES_PATH = "inpictures/" #Must be with '/' at the end
OUT_IMAGES_PATH = "outpictures/" #Must be with '/' at the end
RESIZE_SIZE = (244,244)

MEAN = []
STD = []
torch.manual_seed(1525)


class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return T.functional.rotate(x, angle)


class AddGaussianNoise(object): #this is dangerous do not use!!!
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())  * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MyDataset(Dataset):
  def __init__(self, imgpath_list):
    super(MyDataset, self).__init__()
    for imgpath in imgpath_list:
        pimage = Image.open(IN_IMAGES_PATH + imgpath)
        self.img_list.append(pimage)

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    img = self.img_list[idx]
    return self.img



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
            axs[row_idx, col_idx].imshow( imgs[img_idx] )
            axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[] )
            img_idx += 1

    plt.tight_layout()
    #plt.figure(figsize=(20, 20))
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

        sub_img = my_Transform(pimage)
        img_normalized = np.array(sub_img)
        img_normalized = img_normalized.transpose(1, 2, 0)
        plt.imshow(img_normalized)
        plt.show()
        img_normalized = img_normalized
        img_normalized = img_normalized.astype(np.uint8)
        sub_img = Image.fromarray(img_normalized )

        #plt.xticks([])
        #plt.yticks([])

        #images_np = sub_img.numpy()
        #img_plt = images_np.transpose(0, 2, 3, 1)
        # display 5th image from dataset
        #plt.imshow(img_plt[4])


        # Rotate
        #resimg = my_Transform(pimage)
        # Compress
        #if pimage.size[0] > RESIZE_SIZE[0] and pimage.size[1] > RESIZE_SIZE[1]:
        #    resimg = resimg.resize(RESIZE_SIZE)

        # Save
        sub_img.save(OUT_IMAGES_PATH + image)
        # Set old modification time
        os.utime(OUT_IMAGES_PATH + image, (ti_m, ti_m))
    #sub_imgs = [my_Transform(pimage) for _ in range(4)]
    #plot(pimage, sub_imgs)
    #print("Koniec")


def calculateMeanStd():
    # calcuate means and stds:
    # dataset = MyDataset(image_list)
    dataset = datasets.ImageFolder(IN_IMAGES_PATH[:-1], transform=T.ToTensor())
    # loader = DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False)
    mean = 0.0
    std = 0.0
    for img, _ in dataset:
        mean += img.mean([1, 2])
        std += img.std([1, 2])
    mean /= len(dataset)
    std /= len(dataset)
    print("Dataset mean = " + str(mean))
    print("Dataset std =  " + str(std))
    MEAN = mean.tolist()
    STD = std.tolist()
    return MEAN, STD
    #Train set:
    #[0.5049, 0.4650, 0.4300]
    #[0.1872, 0.1798, 0.1829]

def fun():
    return calculateMeanStd()[0]


def fun2():
    return calculateMeanStd()[1]


my_Transform = T.Compose([
    #T.Resize(RESIZE_  SIZE),
    #T.RandomVerticalFlip(p=0.3),
    #T.RandomInvert(p=0.1),
    #T.RandomHorizontalFlip(p=0.3),
    #T.RandomEqualize(p=0.),
    T.transforms.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],#[0.5049, 0.4650, 0.4300], #[0.5125,0.4667,0.4110], #[0.5049, 0.4650, 0.4300],
                std=[0.229, 0.224, 0.225]),#[0.1872, 0.1798, 0.1829]), #[0.2621,0.2501,0.2453]), #[0.1872, 0.1798, 0.1829]),

    #T.transforms.RandomApply(
    #        [AddGaussianNoise(0., 0.01)],
    #        p=0.
    #    ),
    #T.transforms.GaussianBlur((5,9),(1,5)),
    #T.transforms.ToPILImage(),

    #T.RandomGrayscale(p=0.1),
    #T.transforms.RandomApply(
    #    [T.RandomRotation(degrees=(0, 360))],p=0.5
    #)


])





if __name__ == '__main__':
    main()