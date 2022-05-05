from PIL import Image, ImageStat, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
import os
import utilities
import random

IN_IMAGES_PATH = "inpictures/" #Must be with '/' at the end
OUT_IMAGES_PATH = "outpictures/" #Must be with '/' at the end
RESIZE_SIZE = (244,244)

MEAN = []
STD = []

RANDOM_SEED = True

seed = int(random.random()*100 * random.random()) if RANDOM_SEED else 1525

torch.manual_seed(seed)


NEW_IMAGE_CREATION_MODE = False

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


class EnhanceBrightness(object): #Karol's work
    '''This is custom transform class
        It creates bright circle in the center of image
        when initializing you can set
        brightness - float value determine minimal brightness would be product image, should be in <0, 1) <- dimmer <1 , max_bright) <- lighter
        probability - float value (0,1> determines the logically of preforming the transformation
        max_bright - float value (not smaller than bright) maximal brightness of the product image
        when you call it for specific picture it performs call method. Original design Karol Dziki, altered by ATM'''

    def __init__(self, bright :float = 2.5,  max_bright: float = 3.0, probability : float = 1.0):
        if bright < 0 : return 1
        if max_bright < bright : return 1
        self.max_bright :float = max_bright
        self.bright : float = bright
        self.probability : float =  probability

    def __call__(self, img):
        fate = random.random() #rand number from (0,1>
        fate_bright = random.random() * abs(self.max_bright - self.bright)  + self.bright # rand number from (0,1>
        if fate <= self.probability:
            return ImageEnhance.Brightness(img).enhance(fate_bright)
        else:
            return img # do nothing



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

    if NEW_IMAGE_CREATION_MODE is False:

        # Perform Computations:
        for image in image_list:
            ti_m = os.path.getmtime(IN_IMAGES_PATH + image)
            # Open
            pimage = Image.open(IN_IMAGES_PATH + image)
            pimage = ImageOps.exif_transpose(pimage)

            sub_img = my_Transform(pimage)
            '''
            #img_normalized = np.array(sub_img)
            #img_normalized = img_normalized.transpose(1, 2, 0)
            #plt.imshow(img_normalized)
            #plt.show()
            #img_normalized = img_normalized
            #img_normalized = img_normalized.astype(np.uint8)
            #sub_img = Image.fromarray(img_normalized )
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
            '''
            # Save
            sub_img.save(OUT_IMAGES_PATH + image)
            # Set old modification time
            os.utime(OUT_IMAGES_PATH + image, (ti_m, ti_m))
            sub_imgs = [my_Transform(pimage) for _ in range(4)]
            plot(pimage, sub_imgs)
        print("Koniec")

    else:
        with open("out_LipenLabel.csv", "w", encoding='utf-8') as out_file:
            for image in image_list:
                with open("UniformDatasetLabel.csv", "r", encoding='utf-8') as file:
                    for label in file:
                        name = label.split(";")[0]
                        if name == "\n": continue
                        if name == "Name": continue
                        if name != image: continue

                        # image matching with label:
                        tag = label.split(";")[1]
                        subclass = label.split(";")[2]
                        extra = label.split(";")[3]
                        author = label.split(";")[4]
                        ending = "\n"
                        ti_m = os.path.getmtime(IN_IMAGES_PATH + image)
                        pimage = Image.open(IN_IMAGES_PATH + image)
                        pimage = ImageOps.exif_transpose(pimage)
                        sub_imgs = [my_Transform(pimage) for _ in range(2)]
                        for i,sub_img in enumerate(sub_imgs):
                            # Save
                            path = image.split(".")[0]+"_"+str(i)+"."+image.split(".")[1] #works if path does not have "." inside
                            sub_img.save(OUT_IMAGES_PATH + path)
                            # Set old modification time
                            os.utime(OUT_IMAGES_PATH + path, (ti_m, ti_m))
                            newline = ";".join([path, tag, subclass, extra, author, ending])
                            out_file.write(newline)





my_Transform = T.Compose([
    T.Resize(RESIZE_SIZE),  ## 244p

    T.RandomVerticalFlip(p=0.3),
    T.RandomHorizontalFlip(p=0.3),
    T.transforms.RandomApply(
        [T.transforms.ColorJitter(brightness=(0.9, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=(-0.5, 0.5))],
        p=0.5),
    T.transforms.ToTensor(),

    #T.Normalize(mean=[0.4784, 0.4712, 0.4662],
    #            std=[0.2442, 0.2469, 0.2409]),
    T.transforms.RandomApply(
            [AddGaussianNoise(0., 0.005)],
            p=0.41
        ),
    T.transforms.RandomApply(
        [T.transforms.GaussianBlur((1,9),(0.1,5))],
        p = 0.3),

    T.transforms.ToPILImage(),
    EnhanceBrightness(bright=1.1,max_bright=1.6,probability=0.2),
    T.RandomInvert(p=0.1),
    T.RandomEqualize(p=0.3),
    T.RandomGrayscale(p=0.1),
    T.transforms.RandomApply(
        [T.RandomRotation(degrees=(0, 360))], p=0.5
    ),





])





if __name__ == '__main__':
    main()