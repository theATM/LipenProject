import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_

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
        multi_batch_loss = 0.0
        grad_per_batch_const = 8 #TODO
        for i, data in enumerate(train_loader):
            inputs = torch.autograd.Variable(data['image'].to(device, non_blocking=True))
            labels = torch.autograd.Variable(data['label'].to(device, non_blocking=True))
            with torch.set_grad_enabled(True):
                # Calculate Network Function (what Network thinks of this image)
                output = model(inputs)
                # Calculate loss
                loss = criterion(output, labels)
                # Backpropagate loss
                loss.backward()
                multi_batch_loss += loss;
                if ((i + 1) % multi_batch_loss) == 0:
                    # Normalize loss to account for batch accumulation
                    multi_batch_loss = multi_batch_loss / multi_batch_loss
                    # Clipping the gradient
                    clip_grad_norm_(model.parameters(), max_norm=1)
                    # Update the weighs
                    optimizer.step()
                    # Resets Gradient to Zeros (clearing it before using it next time in calculations)
                    optimizer.zero_grad()
                    # Resets multi batch loss sum
                    multi_batch_loss = 0.0
                # Calculate Accuracy
                # Update Statistics
                # TODO
        # Print Results per epoch
        # TODO
        # Stepping scheduler
        scheduler.step()

        # Evaluate in some epochs:
        epoch_per_eval_const = 10 #TODO
        if (epoch + 1) % epoch_per_eval_const == 0:
            model.eval()
            evaluation_time = time.perf_counter()
            # Evaluate
            # TODO
            # Save Model Checkpoint
            # TODO
        print('Epoch ' + str(epoch + 1) + ' completed')

    print("\nTraining concluded\n")
    model.eval()
    # Post Training Evaluation
    # TODO
    # Print Stats
    # TODO
    # Save Model
    # TODO
    print("Bye")




if __name__ == "__main__":
    main()
