import os
import random
import time
from PIL import Image , ImageOps
import torchvision

'''
This script will help you compress images to 244 x 244
And will rotate them randomly by n*90 degrees
'''


IN_IMAGES_PATH = "inpictures/" #Must be with '/' at the end
OUT_IMAGES_PATH = "outpictures/" #Must be with '/' at the end
RESIZE_SIZE = (244,244)

class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return torchvision.transforms.functional.rotate(x, angle)




def main():
    print("Witaj w programie kompresującym i obracającym otagowane już zdjęcia")
    #Check imput
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

    image_list = getFiles(IN_IMAGES_PATH)
    image_amount = len(image_list)
    if image_amount == 0:
        print("Nie znaleziono żadnych zdjęć ")
        return 1

    # Perform Computations:
    for image in image_list:
        ti_m = os.path.getmtime(IN_IMAGES_PATH + image)
        #Open
        pimage = Image.open(IN_IMAGES_PATH + image)
        pimage = ImageOps.exif_transpose(pimage)
        #Rotate
        resimg = my_Transform(pimage)
        #Compress
        if pimage.size[0] > RESIZE_SIZE[0] and pimage.size[1] > RESIZE_SIZE[1]:
            resimg = resimg.resize(RESIZE_SIZE)
        #Save
        resimg.save(OUT_IMAGES_PATH + image)
        #Set old modification time
        os.utime(OUT_IMAGES_PATH + image,(ti_m,ti_m))

    print("Koniec")





def getFiles(dir):
    image_list = []
    if dir[-1] != '/': dir = dir + '/'

    # Create subfolders
    if dir != IN_IMAGES_PATH and dir != IN_IMAGES_PATH[-1]:
        sdir = dir
        if dir.split('/')[0] + '/' == IN_IMAGES_PATH:
            sdir = "/".join(dir.split('/')[1:])  # do not save image path to image name
        if not os.path.isdir(OUT_IMAGES_PATH + sdir):
            os.mkdir(OUT_IMAGES_PATH + sdir)

    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            image_list.extend(getFiles(dir + file))
        if os.path.isfile(os.path.join(dir, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
            if dir.split('/')[0]+'/' == IN_IMAGES_PATH:
                sdir = "/".join(dir.split('/')[1:]) # do not save image path to image name
            image_list.append(str(sdir + file))
    return image_list





#Here you can set up any transforms you wish:
my_Transform = torchvision.transforms.Compose([
    RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
])





if __name__ == '__main__':
    main()