import os
import time
from PIL import Image , ImageOps
import utilities


AUTHOR_TAG = 'atm' #SELECT YOUR OWN TAG  - MAX 3 DIGIT
IN_IMAGES_PATH = "inpictures/" #Must be with '/' at the end
OUT_IMAGES_PATH = "outpictures/" #Must be with '/' at the end
RESIZE_SIZE = (432,432)


def main():
    #Check if imput is vaild
    if AUTHOR_TAG == '' :
        print("Dodaj tag autora")
        return 1
    if not os.path.isdir(IN_IMAGES_PATH):
        print("Nie znaleziono filderu ze zdjęciami")
        print("Utwórz folder \"" + IN_IMAGES_PATH[:-1]+ "\"")
        return 1
    if IN_IMAGES_PATH[-1] != '/' or  OUT_IMAGES_PATH[-1] != '/':
        print("Dodaj znaki \/ na koniec nazw folderów ze zdjęciami")
        return 1
    #Create out dir
    if not os.path.isdir(OUT_IMAGES_PATH):
        os.mkdir(str(OUT_IMAGES_PATH))
    #Get Images:
    image_list = utilities.getFiles(IN_IMAGES_PATH,IN_IMAGES_PATH,OUT_IMAGES_PATH,True)
    image_amount = len(image_list)
    if image_amount == 0:
        print("Nie znaleziono żadnych zdjęć ")
        return 1
    #Perform Computations:
    for image in image_list:
        ti_m = os.path.getmtime(IN_IMAGES_PATH + image)
        pimage = Image.open(IN_IMAGES_PATH + image)
        pimage = ImageOps.exif_transpose(pimage)
        #add author sufix
        image = image[::-1].split('.')[-1][::-1] + "_" + str(AUTHOR_TAG) + "."+ image[::-1].split('.')[0][::-1]
        if pimage.size[0] > RESIZE_SIZE[0] and pimage.size[1] > RESIZE_SIZE[1]:
            #Resize image
            pimage = pimage.resize(RESIZE_SIZE)
            #image = image.split('.')[0] + ".jpg" # To save as jpg
            #pimage = pimage.convert('RGB')
            pimage.save(OUT_IMAGES_PATH + image)
            os.utime(OUT_IMAGES_PATH + image,(ti_m,ti_m))
    print("Added author tag")
    print("Changed images resolution")








if __name__ == '__main__':
    main()