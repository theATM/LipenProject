import random
import sys
import time
import numpy as np
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
import matplotlib.pyplot as plt


def main():
    # Measure training time
    start_time = time.perf_counter()
    # Load Parameters
    hparams : pl.Hparams = pl.loadProfile(sys.argv)
    # Set proper device
    train_device = torch.device(hparams['train_device'].value)
    if train_device == 'cuda': torch.cuda.empty_cache()    #Empty GPU Cache before Training starts
    # Set initial seed
    seed = 1410
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    hparams['train_seed'] = seed
    # Load Data
    train_loader, val_loader, _ = dl.loadData(hparams,load_train=True,load_val=True)
    # Pick Model
    model = ml.pickModel(hparams).to(train_device)
    # Pick Other Elements
    criterion = ml.pickCriterion(hparams,en.CriterionPurpose.TrainCriterion)
    val_criterion = ml.pickCriterion(hparams, en.CriterionPurpose.EvalCriterion)
    optimizer = ml.pickOptimizer(model,hparams)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hparams['scheduler_list'],gamma=hparams["scheduler_gamma"])
    # Add tensorboard writer (use it with (in terminal): tensorboard --logdir=Logs/Runs)
    writer = SummaryWriter("Logs/Runs/"+ml.getModelName(hparams,withdataset=True))
    # Declare range of epochs to iterate
    max_epoch = hparams['max_epoch']
    min_epoch = 0
    #Set grad per batch
    grad_per_batch = hparams["grad_per_batch"]
    # Set epoch per eval
    epoch_per_eval = hparams["epoch_per_eval"]
    # choose save model mode
    save_mode = hparams["save_mode"]
    # Get Criterion weights
    class_weights = torch.tensor(hparams[hparams['dataset_name'].value + '_class_weights']).to(train_device)
    # Get criterion reductin mode
    reduction_mode =  hparams['reduction_mode']
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
    single_batch_test=hparams['single_batch_test']
    #Run training
    results = train(hparams=hparams,train_device=train_device,
                    train_loader=train_loader,val_loader=val_loader,
                    model=model,criterion=criterion,val_criterion=val_criterion,optimizer=optimizer,scheduler=scheduler,
                    min_epoch=min_epoch,max_epoch=max_epoch,epoch_per_eval=epoch_per_eval,grad_per_batch=grad_per_batch,
                    single_batch_test=single_batch_test, writer = writer, class_weights=class_weights,reduction_mode=reduction_mode,
                    interactive=True,save_mode=save_mode)


def train(
            hparams : pl.Hparams, train_device : str,
            train_loader, val_loader,
            model, criterion,val_criterion, optimizer, scheduler,
            min_epoch, max_epoch, epoch_per_eval, grad_per_batch,
            single_batch_test : bool, writer, class_weights,reduction_mode,
            interactive: bool, save_mode : en.SavingMode

         ):

    # Track best acc for smarter savepoint creation
    best_acc = 0
    # Measure training time
    train_time = time.perf_counter()
    # Start Training
    for epoch in range(min_epoch,max_epoch):

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
            image_dims = (data.size()[0], int(data[0][2]), int(data[0][3]), int(data[0][4]))
            image = np.reshape(data[:, 5:], image_dims)
            inputs = torch.autograd.Variable(image.to(train_device, non_blocking=True))
            labels = torch.autograd.Variable(data[:, 0].long().to(train_device, non_blocking=True))
            weights = torch.autograd.Variable(data[:, 1].to(train_device, non_blocking=True))
            with torch.set_grad_enabled(True):
                # Calculate Network Function (what Network thinks of this image)
                outputs = model(inputs)
                # Calculate loss
                if reduction_mode == en.ReductionMode.none:
                    #Load extras:
                    intermediate_losses = criterion(outputs, labels)
                    loss = torch.mean(weights * intermediate_losses)
                    train_loader.dataset.images[i*image_dims[0]:(i+1)*image_dims[0], 2] = eva.weightChange(outputs, labels, weights).cpu()
                else:
                    loss = criterion(outputs, labels)
                # Normalize loss to account for batch accumulation
                multi_batch_loss = loss / grad_per_batch
                # Back propagate loss
                multi_batch_loss.backward()
                # Calculate, minding gradient batch accumulation
                if ((i + 1) % grad_per_batch) == 0 or single_batch_test is True:
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
                if writer is not None:
                    writer.add_scalar("Loss/train",avg_loss.avg , epoch)
                    writer.add_scalar("Accuracy/train", acc.avg, epoch)
                    writer.add_scalar("Top2Acc/train", acc2.avg, epoch)
                    writer.add_scalar("Top3Acc/train", acc3.avg, epoch)
        # Print Result for One Epoch of Training
        if interactive:
            print(end="\n")
            print('Train | Epoch, {epoch:d} |  *  | Learning rate, {learn_rate:.3e}  | Used Time, {epoch_time:.2f} s |'
                  .format(epoch=epoch,learn_rate=scheduler.get_last_lr().pop(),epoch_time=time.perf_counter() - epoch_time))
            print('Train | Epoch, {epoch:d} |  *  | Loss, {avg_loss.avg:.3f} | Accuracy, {acc.avg:.3f} | In Top 2, {acc2.avg:.3f} | In Top 3, {acc3.avg:.3f} | '
                  .format(epoch=epoch,acc=acc, acc2=acc2, acc3=acc3, avg_loss=avg_loss))
        # Stepping scheduler
        scheduler.step()

        # Evaluate in some epochs:

        if epoch % epoch_per_eval == 0 :
            model.eval()
            evaluation_time = time.perf_counter()
            # Evaluate on valset
            loss_val, (acc_val, acc2_val, acc3_val), precision, recall, f1_score, conf_matrix, roc_auc_avg, roc_fig = \
                eva.evaluate(model,val_criterion,val_loader,train_device, hparams, reduction_mode)
            if train_device == 'cuda:0': torch.cuda.empty_cache()
            # Save Model Checkpoint
            model_saved :bool = False
            if save_mode != en.SavingMode.none_save and save_mode != en.SavingMode.last_save:
                if save_mode == en.SavingMode.all_save or (save_mode == en.SavingMode.best_save and best_acc >= acc_val.avg):
                    best_acc = acc_val.avg if best_acc >= acc_val.avg else best_acc
                    save_params = {"current_epoch":epoch,"current_acc":acc_val.avg,"current_loss":loss_val.avg}
                    ml.saveModel(model,optimizer,scheduler,hparams,save_params)
                    model_saved = True

            # Record Statistics
            if writer is not None:
                writer.add_scalar("Loss/eval", loss_val.avg, epoch)
                writer.add_scalar("Accuracy/eval", acc_val.avg, epoch)
                writer.add_scalar("Top2Acc/eval", acc2_val.avg, epoch)
                writer.add_scalar("Top3Acc/eval", acc3_val.avg, epoch)
                writer.add_scalar("Precision/eval", precision, epoch)
                writer.add_scalar("Recall/eval", recall, epoch)
                writer.add_scalar("F1 Score/eval", f1_score, epoch)
                if not sys.gettrace():
                    writer.add_scalar("Average AUC-ROC", roc_auc_avg, epoch)
                    writer.add_figure("Confusion matrix", conf_matrix, epoch)
                    writer.add_figure("ROC", roc_fig, epoch)
                    plt.close('all')
            # Print Statistics
            if interactive:
                print('Eval  | Epoch, {epoch:d} |  #  | Saved, {model_saved:s} | Used Time, {epoch_time:.2f} s |'
                      .format(epoch=epoch, model_saved=str(model_saved),  epoch_time=time.perf_counter() - evaluation_time))
                print('Eval  | Epoch, {epoch:d} |  #  | Loss, {loss:.3f} | Accuracy, {acc:.3f} | In Top 2, {acc2:.3f} | In Top 3, {acc3:.3f} |'
                      .format(epoch=epoch, loss=loss_val.avg.item(), acc=acc_val.avg.item(), acc2=acc2_val.avg.item(),  acc3=acc3_val.avg.item()))
                if not sys.gettrace():
                    print('Eval  | Epoch, {epoch:d} |  #  | Precision, {precision: .3f} | Recall, {recall: .3f} | F1 Score, {f1_score: .3f} | Avg. AUC-ROC, {aucroc: .3f} |'
                          .format(epoch=epoch, precision=precision.item(), recall=recall.item(), f1_score=f1_score.item(), aucroc=roc_auc_avg.item()))
                else:
                    print('Eval  | Epoch, {epoch:d} |  #  | Precision, {precision: .3f} | Recall, {recall: .3f} | F1 Score, {f1_score: .3f}'
                          .format(epoch=epoch, precision=precision.item(), recall=recall.item(), f1_score=f1_score.item()))
        train_loader.dataset.shuffle_images()

    model.eval()
    # Post Training Evaluation on valset (for comparisons)
    vloss_avg, (vacc_avg, vacc2_avg, vacc3_avg), precision, recall, f1_score, conf_matrix, roc_auc_avg, roc_fig = \
        eva.evaluate(model, val_criterion, val_loader, train_device, hparams, reduction_mode)
    # Post Training Evaluation on testset (for true accuracy) - do not do that
    # tloss_avg, (tacc_avg, tacc2_avg, tacc3_avg) = eva.evaluate(model,criterion,test_loader,train_device,hparams)
    if interactive:
        #Print results on eval set
        print("\nTraining concluded\n")
        print("Evaluation on validation set")
        print('Evaluation accuracy at the end on all validation images, %2.2f' % vacc_avg.avg.item())
        print('Top 2 at the end on all validation images, %2.2f' % vacc2_avg.avg.item())
        print('Top 3 at the end on all validation images, %2.2f' % vacc3_avg.avg.item())
        print('Average loss at the end on all validation images, %2.2f' % vloss_avg.avg.item())
        print('Confusion matrix\n:')
        print(conf_matrix)
        # Print results on test set - do not do that
        #print("\nEvaluation on test set")
        #print('Evaluation accuracy on all test images, %2.2f' % tacc_avg.avg)
        #print('Top 2 at the end on all test images, %2.2f' % tacc2_avg.avg)
        #print('Top 3 at the end on all test images, %2.2f' % tacc3_avg.avg)
        #print('Average loss on all test images, %2.2f' % tloss_avg.avg)
        #print("\nFinished Training\n")
    # Save Last Model
    if save_mode != en.SavingMode.none_save:
        save_params = {"current_epoch": max_epoch, "current_acc": vacc_avg.avg, "current_loss": vloss_avg}
        ml.saveModel(model, optimizer, scheduler, hparams, save_params)
    if interactive:
        print("Saved model on epoch %d at the end ot training" % max_epoch)
        print("Whole training took %f.2 s",time.perf_counter() - train_time)
        print("Bye")

    return vloss_avg, (vacc_avg, vacc2_avg, vacc3_avg)


if __name__ == "__main__":
    main()
