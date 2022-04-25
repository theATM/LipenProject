import os
import utilities
import protocol
import shutil

IN_PICTURES_PATH = "inpictures/"
OUT_PICTURES_PATH = "outpictures/"

IN_CSV = "JulLipenLabel1n3.csv"
OUT_CSV = "JulLipenLabel1n3Hard.csv"


def main():
    # Check if imput is vaild
    if IN_PICTURES_PATH[-1] != '/' or OUT_PICTURES_PATH[-1] != '/':
        print("Dodaj znaki \/ na koniec nazw folderów ze zdjęciami")
        return 1
    if not os.path.isdir(IN_PICTURES_PATH):
        print("Nie znaleziono filderu ze zdjęciami")
        print("Utwórz folder \"" + IN_PICTURES_PATH[:-1] + "\"")
        return 1
    # Create out dir
    if not os.path.isdir(OUT_PICTURES_PATH):
        os.mkdir(str(OUT_PICTURES_PATH))
    # Get Images:
    image_list = utilities.getFiles(IN_PICTURES_PATH, IN_PICTURES_PATH, OUT_PICTURES_PATH, True)
    image_amount = len(image_list)
    if image_amount == 0:
        print("Nie znaleziono żadnych zdjęć ")
        return 1

    with open(OUT_CSV, "w", encoding='utf-8') as out_file:
        with open(IN_CSV, "r", encoding='utf-8') as in_file:
            for line in in_file:
                name = line.split(";")[0]
                if name == "\n": continue
                if name == "Name":
                    out_file.write(line)
                    continue

                tag = line.split(";")[1]
                subclass = line.split(";")[2]
                extra = line.split(";")[3]
                author = line.split(";")[4]

                #check if image has extra tag -> hard
                if int(extra) & protocol.extralabels["hard"] == protocol.extralabels["hard"]:
                    #save to out file
                    out_file.write(line)
                    #save image to out folder
                    shutil.copyfile(IN_PICTURES_PATH + name, OUT_PICTURES_PATH + name)


    #remove empty dirs:
    for dir in os.listdir(OUT_PICTURES_PATH):
        full_path_dir = os.path.join(OUT_PICTURES_PATH, dir)
        if os.path.isdir(full_path_dir) :
            if len(os.listdir(full_path_dir)) == 0 :
                os.rmdir(full_path_dir)

if __name__ == '__main__':
    main()