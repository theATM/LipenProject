import os
import sys
import random
import shutil
import torch
import Code.Profile.profileloader as pl
import Code.Protocol.enums as en
from Code.Dataloader.lipenset import Lipenset
from torchvision.utils import save_image
from Code.Functional.mean_std import calculate_mean_std
import Code.Dataloader.datatools as dt


def main():
    seed = 1410
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    hparams = pl.loadProfile(sys.argv)
    trainset = Lipenset(hparams, en.DatasetType.Trainset, shuffle=True)

    augmentation_count = hparams["augmentation_count"]

    in_dir = hparams['data_dir'] + "/" + hparams['dataset_dir'] + "/"
    in_dir_images = in_dir + hparams["trainset_dir"] + "/"
    out_dir = in_dir[:-1] + "Augmented/"
    out_dir_images = out_dir + hparams["trainset_dir"] + "/"
    if not os.path.exists(out_dir_images):
        os.makedirs(out_dir_images)
    for subdir in [hparams["trainset_dir"], hparams["valset_dir"], hparams["testset_dir"]]:
        shutil.copytree(in_dir + subdir, out_dir+ subdir, dirs_exist_ok=True)
    labels_file_name = hparams["label_filename"]
    with open(in_dir + labels_file_name, 'r') as csv_file:
        csv_lines = csv_file.readlines()
        csv_header = csv_lines[0]
        csv_labels_raw = csv_lines[1:]
    train_image_files = dt.getImageFiles(in_dir_images)
    csv_labels = {}
    with open(out_dir + labels_file_name.split('.')[0] + "Augmented.csv", 'w') as new_csv_file:
        new_csv_file.write(csv_header)
        for csv_label_raw in csv_labels_raw:
            new_csv_file.write(csv_label_raw)
            full_path = in_dir_images + csv_label_raw.split(';')[0]
            if full_path in train_image_files:
                csv_labels[full_path] = csv_label_raw
        new_csv_file.write('\n')
        for i in range(1, augmentation_count+1):
            for imageDict in trainset:
                csv_line = csv_labels[imageDict["path"]]
                csv_line_split = csv_line.split(';')
                image_file_path_split = csv_line_split[0].split('.')
                new_file_subpath = f"{image_file_path_split[0]}_{i}.{image_file_path_split[1]}"
                csv_line = ';'.join([new_file_subpath] + csv_line_split[1:])
                if csv_line[-1] != '\n':
                    csv_line += '\n'
                new_csv_file.write(csv_line)
                new_file_path = out_dir_images + new_file_subpath
                save_image(imageDict["image"], new_file_path)
            print(f"Augmentation {i}/{augmentation_count} done.")
    calculate_mean_std(hparams)
    print("Mean and std calculated. Exiting...")


if __name__ == "__main__":
    main()
