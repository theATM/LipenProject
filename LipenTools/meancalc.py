from torchvision import datasets
import torchvision.transforms as T
import math

IN_IMAGES_PATH = "inpictures/"


def main():
    calculateMeanStd()
    return


def calculateMeanStd(in_path):
    # calcuate means and stds:
    dataset = datasets.ImageFolder(IN_IMAGES_PATH[:-1], transform=T.ToTensor())
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
    print("Dataset mean = " + str(mean))
    print("Dataset std =  " + str(std))
    return mean, std


if __name__ == '__main__':
    main()


#Results:

#Relevant:
#Clean set (train):
#mean = [0.5044, 0.4650, 0.4307]
#std = [0.1833, 0.1763, 0.1791]

#Uniform set (train):
#mean = [0.5044, 0.4657, 0.4323]
#std = [0.1849, 0.1782, 0.1817]

#Merged set (train + eval):
#mean = [0.5074, 0.4685, 0.4339]
#std =  [0.1857, 0.1791, 0.1823]

##Old:

#Old Augmented Train set:
#[0.5049, 0.4650, 0.4300]
#[0.1872, 0.1798, 0.1829]


#Old Merged Set:
#mean = tensor([0.5076, 0.4680, 0.4326])
#std = tensor([0.1844, 0.1776, 0.1801])

#Other:
# ([0.4784, 0.4712, 0.4662])
# ([0.2442, 0.2469, 0.2409])
