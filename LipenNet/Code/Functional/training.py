import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

import Code.Profile.profileloader as pl
import Code.Dataloader.lipenset as dl
import Code.Architecture.modelloader as ml


def main():
    hparams : pl.Hparams = pl.loadProfile(sys.argv)
    device = torch.device(hparams['train_device'].value)
    if device == 'cuda': torch.cuda.empty_cache()    #Empty GPU Cache before Training starts
    writer = SummaryWriter("Logs/Runs")
    train_loader, val_loader, test_loader = dl.loadData(hparams)
    model = ml.pickModel(hparams)
    criterion = ml.pickCriterion(hparams)
    optimizer = ml.pickOptimizer(model,hparams)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hparams['train_scheduler_list'],gamma=0.75)

    model.to(device)
    if hparams['train_single_batch_test'] is True:
        # Preform Single Batch Test
        train_loader = [next(iter(train_loader))]
        print("Single Batch Test Chosen")

    max_epoch = hparams['train_max_epoch']
    for epoch in range(max_epoch):
        print('\n','Epoch', epoch+1)
        print("Learning rate = %1.5f" % scheduler.get_last_lr().pop())
        #Train one epoch
        model.train()
        





if __name__ == "__main__":
    main()
