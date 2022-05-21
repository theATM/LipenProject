import os
import sys
from torchvision.utils import save_image
import Code.Profile.profileloader as pl
import Code.Protocol.enums as en
from Code.Dataloader.lipenset import Lipenset
from Code.Dataloader.transforms import LipenTransform

from skimage import io
io.use_plugin('matplotlib')


def main():
    hparams = pl.loadProfile(sys.argv)
    transformer = LipenTransform(en.AugmentationType.Online, hparams)
    trainset = Lipenset(hparams, en.DatasetType.Trainset, shuffle=True)
    outDir = hparams['data_dir'] + "/" + hparams['dataset_dir'] + "Augmented/" + hparams["trainset_dir"] + "/"

    for imageDict in trainset:
        image = imageDict["image"]
        image_augmented = transformer.transform(image)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        save_image(image_augmented, outDir + os.path.basename(imageDict["path"]))

if __name__ == "__main__":
    main()