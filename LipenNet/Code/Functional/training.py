import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_

import Code.Profile.profileloader as pl
import Code.Dataloader.lipenset as dl
import Code.Architecture.modelloader as ml
import Code.Functional.utilities as ut
import Code.Functional.evaluation as eva
import Code.Protocol.enums as en


def main():
    # Measure training time
    start_time = time.perf_counter()
    # Load Parameters
    hparams : pl.Hparams = pl.loadProfile(sys.argv)
    # Set proper device
    train_device = torch.device(hparams['train_device'].value)
    if train_device == 'cuda': torch.cuda.empty_cache()    #Empty GPU Cache before Training starts
    # Load Data
    train_loader, val_loader, test_loader = dl.loadData(hparams)
    # Pick Model
    model = ml.pickModel(hparams).to(train_device)
    # Pick Other Elements
    criterion = ml.pickCriterion(hparams)
    optimizer = ml.pickOptimizer(model,hparams)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hparams['scheduler_list'],gamma=hparams["scheduler_gamma"])
    # Add tensorboard writer
    writer = SummaryWriter("Logs/Runs") #TODO make use of it
    # Declare range of epochs to iterate
    max_epoch = hparams['max_epoch']
    min_epoch = 0
    #Set grad per batch
    grad_per_batch = hparams["grad_per_batch"]
    # Set epoch per eval
    epoch_per_eval = hparams["epoch_per_eval"]
    # choose save model mode
    save_mode = hparams["save_mode"]
    # Prepare Save dir
    ml.savePrepareDir(hparams)
    # Load model if required
    if hparams['load_model']:
        load_params = ml.loadModel(model,optimizer,scheduler,train_device,hparams)
        if "min_epoch" in load_params:
            min_epoch = load_params["current_epoch"]

    # Alternative single batch test (to check if network will over fit)
    if hparams['single_batch_test'] is True:
        # Preform Single Batch Test
        train_loader = [next(iter(train_loader))]
        print("Single Batch Test Chosen")


def train(
            hparams : pl.Hparams, train_device : str,
            train_loader, val_loader, test_loader,
            model, criterion, optimizer, scheduler,
            min_epoch, max_epoch, epoch_per_eval, grad_per_batch,
            single_batch_test : bool, writer,
            interactive: bool, save_mode : en.SavingMode

         ):

    # Track best acc for smarter savepoint creation
    best_acc = 0
    # Measure training time
    train_time = time.perf_counter()

    # Start Training
    for epoch in range(min_epoch,max_epoch):
        if interactive:
            print('\nEpoch', epoch)
            print("Learning rate = %1.5f" % scheduler.get_last_lr().pop())
        # Defines statistical variables
        acc = ut.AverageMeter('Accuracy', ':6.2f')
        acc2 = ut.AverageMeter('Top 2 Accuracy', ':6.2f')
        acc3 = ut.AverageMeter('Top 3 Accuracy', ':6.2f')
        avg_loss = ut.AverageMeter('Loss', '1.5f')
        # define useful variables
        multi_batch_loss = 0.0

        # Measure one epoch time
        epoch_time = time.perf_counter()
        # Train one epoch
        model.train()
        for i, data in enumerate(train_loader):
            inputs = torch.autograd.Variable(data['image'].to(train_device, non_blocking=True))
            labels = torch.autograd.Variable(data['label'].to(train_device, non_blocking=True))
            with torch.set_grad_enabled(True):
                # Calculate Network Function (what Network thinks of this image)
                outputs = model(inputs)
                # Calculate loss
                loss = criterion(outputs, labels)
                # Back propagate loss
                loss.backward()
                multi_batch_loss += loss
                # Calculate, minding gradient batch accumulation
                if ((i + 1) % grad_per_batch) == 0 or single_batch_test is True:
                    # Normalize loss to account for batch accumulation
                    multi_batch_loss = multi_batch_loss / grad_per_batch
                    # Clipping the gradient
                    clip_grad_norm_(model.parameters(), max_norm=1)
                    # Update the weighs
                    optimizer.step()
                    # Resets Gradient to Zeros (clearing it before using it next time in calculations)
                    optimizer.zero_grad()
                    # Resets multi batch loss sum
                    multi_batch_loss = 0.0
                # Calculate Accuracy
                c_acc, c_acc2, c_acc3 = eva.accuracy(outputs,labels,topk=(1,2,3))
                # Update Statistics
                acc.update(c_acc[0],inputs.size(0))
                acc2.update(c_acc2[0], inputs.size(0))
                acc3.update(c_acc3[0], inputs.size(0))
                avg_loss.update(loss,inputs.size(0)) #TODO check if loss is correct!
        # Print Result for One Epoch of Training
        if interactive:
            print('Epoch {epoch:d}:  *  | Loss {avg_loss.avg:.3f} | Accuracy {acc.avg:.3f} | In Top 2 {acc2.avg:.3f} | In Top 3 {acc3.avg:.3f} | Used Time {epoch_time:.2f} s'
                  .format(epoch=epoch,acc=acc, acc2=acc2, acc3=acc3, avg_loss=avg_loss, epoch_time=time.perf_counter() - epoch_time))
        # Stepping scheduler
        scheduler.step()

        # Evaluate in some epochs:

        if epoch % epoch_per_eval == 0 and epoch != 0:
            model.eval()
            evaluation_time = time.perf_counter()
            # Evaluate on valset
            loss_avg, (acc_avg, acc2_avg, acc3_avg) = eva.evaluate(model,criterion,val_loader,train_device,hparams) #TODO unpack accuracies
            if train_device == 'cuda:0': torch.cuda.empty_cache()
            # Save Model Checkpoint

           if save_mode != en.SavingMode.none_save and save_mode != en.SavingMode.last_save:
                if save_mode == en.SavingMode.all_save or (save_mode == en.SavingMode.best_save and best_acc >= acc_avg.avg):
                    best_acc = acc_avg.avg if best_acc >= acc_avg.avg else best_acc
                    save_params = {"current_epoch":epoch,"current_acc":acc_avg.avg,"current_loss":loss_avg.avg}
                    ml.saveModel(model,optimizer,scheduler,hparams,save_params)
                    if interactive:
                        print("Saved model on epoch %d" % epoch)

            # Print Statistics
            if interactive:
                print('Evaluation on epoch %d accuracy on all validation images, %2.2f' % (epoch, acc_avg.avg))
                print('Top 2 on epoch %d on all validation images, %2.2f' % (epoch, acc2_avg.avg))
                print('Top 3 on epoch %d on all validation images, %2.2f' % (epoch, acc3_avg.avg))
                print('Average loss on epoch %d on all validation images, %2.2f' % (epoch,loss_avg.avg))
                print('Evaluation on epoch %d took %.2f s' % (epoch, time.perf_counter() - evaluation_time))
        if interactive:
            print('Epoch ' + str(epoch) + ' completed')


    model.eval()
    # Post Training Evaluation on valset (for comparisons)
    vloss_avg, (vacc_avg, vacc2_avg, vacc3_avg) = eva.evaluate(model,criterion,val_loader,train_device,hparams)
    # Post Training Evaluation on testset (for true accuracy)
    tloss_avg, (tacc_avg, tacc2_avg, tacc3_avg) = eva.evaluate(model,criterion,test_loader,train_device,hparams)
    if interactive:
        #Print results on eval set
        print("\nTraining concluded\n")
        print("Evaluation on validation set")
        print('Evaluation accuracy at the end on all validation images, %2.2f' % vacc_avg.avg)
        print('Top 2 at the end on all validation images, %2.2f' % vacc2_avg.avg)
        print('Top 3 at the end on all validation images, %2.2f' % vacc3_avg.avg)
        print('Average loss at the end on all validation images, %2.2f' % vloss_avg.avg)
        # Print results on test set
        print("\nEvaluation on test set")
        print('Evaluation accuracy on all test images, %2.2f' % tacc_avg.avg)
        print('Top 2 at the end on all test images, %2.2f' % tacc2_avg.avg)
        print('Top 3 at the end on all test images, %2.2f' % tacc3_avg.avg)
        print('Average loss on all test images, %2.2f' % tloss_avg.avg)
        print("\nFinished Training\n")
    # Save Last Model
    if save_mode != en.SavingMode.none_save:
        save_params = {"current_epoch": max_epoch, "current_acc": vacc_avg, "current_loss": vloss_avg}
        ml.saveModel(model, optimizer, scheduler, hparams, save_params)
    if interactive:
        print("Saved model on epoch %d at the end ot training" % max_epoch)
        print("Bye")

    return vloss_avg, (vacc_avg, vacc2_avg, vacc3_avg)




if __name__ == "__main__":
    main()
