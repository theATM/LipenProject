import os
import utilities
import protocol
from operator import add

IN_PICTURES_PATH = "inpictures/"

#inside IN_Pictures directory:
TRAIN_PATH = "train"
EVAL_PATH = "eval"
TEST_PATH = "test"
ANY_PATH = "any"

label_CSV = "AugmentedDatasetLabel.csv"


Author_order = {"jlk": 0 ,"nmz": 1 , "atm": 2}




def checkIn():
    # Check if imput is vaild
    if IN_PICTURES_PATH[-1] != '/':
        print("Dodaj znaki \/ na koniec nazw folderów ze zdjęciami")
        return 1
    if not os.path.isdir(IN_PICTURES_PATH):
        print("Nie znaleziono filderu ze zdjęciami")
        print("Utwórz folder \"" + IN_PICTURES_PATH[:-1] + "\"")
        return 1
    if not os.path.isdir(IN_PICTURES_PATH+TRAIN_PATH):
        print("Nie znaleziono filderu ")
        print("Utwórz folder \"" + TRAIN_PATH )
        return 1
    if not os.path.isdir(IN_PICTURES_PATH+EVAL_PATH):
        print("Nie znaleziono filderu")
        print("Utwórz folder \"" + EVAL_PATH )
        return 1
    if not os.path.isdir(IN_PICTURES_PATH+TEST_PATH):
        print("Nie znaleziono filderu ")
        print("Utwórz folder \"" + TEST_PATH)
        return 1
    if not os.path.isdir(IN_PICTURES_PATH+ANY_PATH):
        print("Nie znaleziono filderu")
        print("Utwórz folder \"" + ANY_PATH )
        return 1
    if not os.path.isfile(label_CSV):
        print("Nie znaleziono pliku")
        print("Utwórz pliku \"" + label_CSV)
        return 1
    return 0


def main():
    if checkIn() != 0:
        return 1
    image_list = utilities.getFiles(IN_PICTURES_PATH, IN_PICTURES_PATH, None, False)
    image_amount = len(image_list)
    if image_amount == 0:
        print("Nie znaleziono żadnych zdjęć ")
        return 1

    all_sub_count = [[0,0],None,None,[0,0,0,0],[0,0,0],[0,0],None]

    train_count = [0,0,0,0,0,0,0]
    train_sub_count = [[0,0],None,None,[0,0,0,0],[0,0,0],[0,0],None]
    train_extra_count = [0,0,0,0,0,0]
    train_author_count = [0,0,0] # jlk ,nmz , atm

    eval_count = [0,0,0,0,0,0,0]
    eval_sub_count = [[0, 0], None, None, [0, 0, 0, 0], [0, 0, 0], [0, 0], None]
    eval_extra_count = [0, 0, 0, 0, 0, 0]
    eval_author_count = [0,0,0]

    test_count = [0,0,0,0,0,0,0]
    test_sub_count = [[0, 0], None, None, [0, 0, 0, 0], [0, 0, 0], [0, 0], None]
    test_extra_count = [0, 0, 0, 0, 0, 0]
    test_author_count = [0, 0, 0]

    any_count = [0,0,0,0,0,0,0]
    any_sub_count = [[0, 0], None, None, [0, 0, 0, 0], [0, 0, 0], [0, 0], None]
    any_extra_count = [0, 0, 0, 0, 0, 0]
    any_author_count = [0, 0, 0]


    for image in image_list:
        dataset = image.split("/")[0]
        image = "/".join(image.split("/")[1:])
        with open(label_CSV, "r", encoding='utf-8') as label_file:
            for label in label_file:
                name = label.split(";")[0]
                if name == "\n": continue
                if name == "Name": continue
                if name != image : continue

                # image matching with label:
                tag = label.split(";")[1]
                subclass = label.split(";")[2]
                extra = label.split(";")[3]
                author = label.split(";")[4]

                x = TRAIN_PATH
                match dataset:
                    case dataset if dataset == TRAIN_PATH:
                        train_count[int(tag)] += 1
                        if any_sub_count[int(tag)] is not None:
                            train_sub_count[int(tag)][int(subclass)] += 1
                            all_sub_count[int(tag)][int(subclass)] += 1
                        for i in range(0,len(train_extra_count) - 1):
                            x = int(extra) & list(protocol.extralabels.values())[i]
                            train_extra_count[i] += 1 if x != 0 else 0
                        if int(extra) == 0:
                            train_extra_count[len(train_extra_count) - 1] += 1
                        train_author_count[Author_order[author]] += 1

                    case dataset if dataset == EVAL_PATH:
                        eval_count[int(tag)] += 1
                        if any_sub_count[int(tag)] is not None:
                            eval_sub_count[int(tag)][int(subclass)] += 1
                            all_sub_count[int(tag)][int(subclass)] += 1
                        for i in range(0, len(eval_extra_count) - 1):
                            x = int(extra) & list(protocol.extralabels.values())[i]
                            eval_extra_count[i] += 1 if x != 0 else 0
                        if int(extra) == 0:
                            eval_extra_count[len(eval_extra_count) - 1] += 1
                        eval_author_count[Author_order[author]] += 1

                    case dataset if dataset == TEST_PATH:
                        test_count[int(tag)] += 1
                        if any_sub_count[int(tag)] is not None:
                            test_sub_count[int(tag)][int(subclass)] += 1
                            all_sub_count[int(tag)][int(subclass)] += 1
                        for i in range(0, len(test_extra_count) - 1):
                            x = int(extra) & list(protocol.extralabels.values())[i]
                            test_extra_count[i] += 1 if x != 0 else 0
                        if int(extra) == 0:
                            test_extra_count[len(test_extra_count) - 1] += 1
                        test_author_count[Author_order[author]] += 1

                    case dataset if dataset == ANY_PATH:
                        any_count[int(tag)] += 1
                        if any_sub_count[int(tag)] is not None:
                            any_sub_count[int(tag)][int(subclass)] += 1
                            all_sub_count[int(tag)][int(subclass)] += 1
                        for i in range(0, len(any_extra_count) - 1):
                            x = int(extra) & list(protocol.extralabels.values())[i]
                            any_extra_count[i] += 1 if x != 0 else 0
                        if int(extra) == 0:
                            any_extra_count[len(any_extra_count) - 1] += 1
                        any_author_count[Author_order[author]] += 1

    all_count = list(map(add, list(map(add,train_count,eval_count)) ,list(map(add,test_count,any_count))))
    all_extra_count = list(map(add, list(map(add,train_extra_count,eval_extra_count)) ,list(map(add,test_extra_count,any_extra_count))))
    all_author_count = list(map(add, list(map(add,train_author_count,eval_author_count)) ,list(map(add,test_author_count,any_author_count))))
    #Print all:
    print("==================================================================================================================================================================================================")
    print("         Train Set:                               Eval Set:                               Test Set:                               Any Set:                               All:")
    print("==================================================================================================================================================================================================")
    print("Labels   " +  str(train_count) + "                "
                        + str(eval_count) +"                "
                        + str(test_count)+ "                  "
                        + str(any_count)+ "               "
                        + str(all_count))
    print("==================================================================================================================================================================================================")
    print("Extra:   " + str(train_extra_count) + "                       "
                      + str(eval_extra_count) + "                     "
                      + str(test_extra_count) + "                       "
                      + str(any_extra_count) +  "                   "
                      + str(all_extra_count))
    print("==================================================================================================================================================================================================")
    print("Author:  " + str(train_author_count) + "                                "
                      + str(eval_author_count) + "                              "
                      + str(test_author_count) + "                                "
                      + str(any_author_count) + "                            "
                      + str(all_author_count))

    print("==================================================================================================================================================================================================")
    print("Sublabels:")
    print("Train - " + str(train_sub_count))
    print("Eval  - " + str(eval_sub_count))
    print("Test  - " + str(test_sub_count))
    print("Any   - " + str(any_sub_count))
    print("All   - " + str(all_sub_count))
    print("==================================================================================================================================================================================================")
    print("==================================================================================================================================================================================================")
    print("All images - " + str(sum(all_count)))


if __name__ == '__main__':
    main()


