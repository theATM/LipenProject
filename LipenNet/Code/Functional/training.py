import sys
import torch
from torch.utils.tensorboard import SummaryWriter

import Code.Profile.profileloader as pl
import Code.Dataloader.lipenset as dl


def main():
    hparams : pl.Hparams = pl.loadProfile(sys.argv)
    device = torch.device(hparams['train_device'].value)
    writer = SummaryWriter("Logs/Runs")
    train_loader, val_loader, test_loader = dl.loadData(hparams)
    model = None #TODO
    criterion = None #TODO
    optimizer = None #TODO

    model.to(device)
    if hparams['train_single_batch_test'] is True:
        # Preform Single Batch Test
        train_loader = [next(iter(train_loader))]
        print("Single Batch Test Chosen")

    max_epoch = hparams['train_max_epoch']
    for epoch in range(max_epoch):
        print('Epoch ', epoch+1)


def train():
    pass


def train_one_epoch():
    pass


if __name__ == "__main__":
    main()
