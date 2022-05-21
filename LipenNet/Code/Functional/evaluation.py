import torch
import sys
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix

import Code.Functional.utilities as u
import Code.Dataloader.lipenset as dl
import Code.Profile.profileloader as pl
import Code.Architecture.modelloader as ml
import Code.Functional.utilities as ut


def main():
    hparams: pl.Hparams = pl.loadProfile(sys.argv)
    val_device = torch.device(hparams['val_device'].value)
    # Empty GPU Cache before Training starts
    if val_device == 'cuda': torch.cuda.empty_cache()
    # Load Data
    _ , _,testloader  = dl.loadData(hparams)
    model = ml.pickModel(hparams)
    model.to(val_device)
    criterion = ml.pickCriterion(hparams)

    #Eval
    print("\nEvaluation Started")
    model.eval()
    loss, accuracy = evaluate(model,criterion, testloader,val_device, hparams)
    print('Evaluation Loss on all test images, %2.2f' % (loss.avg))
    print('Evaluation Accuracy on all test images, %2.2f' % (accuracy[0].avg))
    print('Evaluation TOP 2 Accuracy on all test images, %2.2f' % (accuracy[1].avg))
    print('Evaluation TOP 3 Accuracy on all test images, %2.2f' % (accuracy[2].avg))
    print("Evaluation Finished")



def evaluate(model,criterion, data_loader,val_device, hparams: pl.Hparams):
    model.eval()
    acc_avg = ut.AverageMeter('Accuracy', ':6.2f')
    loss_avg = ut.AverageMeter('Loss', ':6.2f')

    acc2_avg = ut.AverageMeter('Top 2 Accuracy', ':6.2f')
    acc3_avg = ut.AverageMeter('Top 3 Accuracy', ':6.2f')
    y_true_all = []
    y_pred_all = []

    class_names = ["Triangle", "Rules", "Rubber", "Pencil", "Pen", "None"]
    class_names_pred = [class_name + ".P" for class_name in class_names]
    classes_count = len(class_names)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = torch.autograd.Variable(data['image'].to(val_device, non_blocking=True))
            labels = torch.autograd.Variable(data['label'].to(val_device, non_blocking=True))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc, acc2 ,acc3 = accuracy(outputs,labels,topk=(1,2,3))
            acc_avg.update(acc[0], inputs.size(0))
            acc2_avg.update(acc2[0], inputs.size(0))
            acc3_avg.update(acc3[0], inputs.size(0))
            loss_avg.update(loss,inputs.size(0))

            _, pred = outputs.topk(1, 1, True, True)
            y_pred_all += pred.tolist()
            y_true_all += labels.tolist()
            #TODO other metrics (Fscore, Confusion Matrix, ROC) and check exisitong ones

    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in class_names],
                         columns=[i for i in class_names_pred])
    #df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    conf_matrix_heatmap = sn.heatmap(df_cm, annot=True).get_figure()

    return loss_avg, (acc_avg,acc2_avg,acc3_avg), conf_matrix_heatmap


def accuracy(outputs, labels , topk=(1,)):
    """Computes the accuracy over the top predictions for the specified values of k"""
    with torch.no_grad(): # disables recalculation of gradients
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.contiguous().view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def precision_micro(outputs, labels):
#     with torch.no_grad():
#         batch_size = labels.size(0)
#         _, pred = outputs.topk(1, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(labels.contiguous().view(1, -1).expand_as(pred))
#         correct_count = int(correct.contiguous().view(-1).float().sum())
#         classes_count = outputs.size(1)
#         false_positives = 0
#         for i in range(classes_count):
#             false_positives +=

if __name__ == '__main__':
    main()
