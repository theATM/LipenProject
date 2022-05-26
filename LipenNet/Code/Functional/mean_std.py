import pickle
import math
import sys
import Code.Profile.profileloader as pl
from torchvision import datasets
import torchvision.transforms as T


def calculate_mean_std(hparams, augmentation=True):
    in_dir = hparams['data_dir'] + "/" + hparams['dataset_dir']
    if augmentation:
        out_dir = in_dir + "Augmented"
    else:
        out_dir = in_dir
    in_dir_training = in_dir + "/" + hparams['trainset_dir']

    dataset = datasets.ImageFolder(in_dir_training, transform=T.ToTensor())
    sum_vec = 0.0
    sum_squares = 0.0
    for img, _ in dataset:
        sum_vec += img.sum([1, 2])
        sum_squares += (img**2).sum([1, 2])
    image_size = dataset[0][0].size()
    pixels_in_image = image_size[1] * image_size[2]
    pixels_in_dataset = pixels_in_image * len(dataset)
    mean = [ch_sum / pixels_in_dataset for ch_sum in sum_vec]
    variance = [(ch_sum_squares - ch_sum**2 / pixels_in_dataset) / pixels_in_dataset
                for ch_sum_squares, ch_sum in zip(sum_squares, sum_vec)]
    std = [math.sqrt(ch_var) for ch_var in variance]
    mean = [m_ch.tolist() for m_ch in mean]

    txt_name = hparams['dataset_name'].name + "Mean"
    if augmentation:
        txt_name += "Augmented"
    with open(f"{out_dir}/{txt_name}.txt", 'w') as mean_file:
        mean_file.write(f"mean = {mean}\n")
        mean_file.write(f"std = {std}")

    pickle_name = hparams['normalization_filename'].split('.')[0]
    if augmentation:
        pickle_name += "Augmented"
    with open(f"{out_dir}/{pickle_name}.pickle", 'wb') as mean_file_pickle:
        pickle.dump((mean, std), mean_file_pickle)


def main():
    hparams = pl.loadProfile(sys.argv)
    calculate_mean_std(hparams, False)


if __name__ == "__main__":
    main()
