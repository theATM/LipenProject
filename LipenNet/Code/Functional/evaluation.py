import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

import Code.Dataloader.lipenset as dl
import Code.Profile.profileloader as pl
import Code.Architecture.modelloader as ml
import Code.Functional.utilities as ut
import Code.Protocol.enums as en


def main():
    if len(sys.argv) < 4:
        print("Add more run arguments")
        sys.exit(-1)
    val_type: en.ValType = en.ValType[sys.argv[1]]
    sys.argv = [sys.argv[0]] + sys.argv[-2:]
    hparams: pl.Hparams = pl.loadProfile(sys.argv)
    val_device = torch.device(hparams['val_device'].value)
    # Empty GPU Cache before Training starts
    if val_device == 'cuda': torch.cuda.empty_cache()

    # Load Data
    match val_type:
        case en.ValType.train:
            dataloader ,_ , _ = dl.loadData(hparams, load_train=True)
        case en.ValType.val:
            _, dataloader, _ = dl.loadData(hparams, load_val=True)
        case en.ValType.test:
            _, _, dataloader = dl.loadData(hparams, load_test=True)

    model = ml.pickModel(hparams)
    model.to(val_device)
    criterion = ml.pickCriterion(hparams,val_device)
    if hparams['load_model']:
        ml.load_model_test(model, val_device, hparams)
    # Add tensorboard writer (use it with (in terminal): tensorboard --logdir=Logs/Eval)
    writer = SummaryWriter("Logs/Eval/" + val_type.value.upper() + "_" + ml.getModelName(hparams, withdataset=True))

    # Evaluate on chosen set
    print("\nEvaluation Started")
    model.eval()
    loss_val, (acc_val, acc2_val, acc3_val), precision, recall, f1_score, conf_matrix, roc_auc_avg, roc_fig = \
        evaluate(model, criterion, dataloader, val_device)
    print('Evaluation Loss on all %s images, %2.2f' % (val_type.value , loss_val.avg.item()))
    print('Evaluation Accuracy on all %s images, %2.2f' % (val_type.value ,acc_val.avg.item()))
    print('Evaluation TOP 2 Accuracy on all %s images, %2.2f' %  (val_type.value ,acc2_val.avg.item()))
    print('Evaluation TOP 3 Accuracy on all %s images, %2.2f' % (val_type.value , acc3_val.avg.item()))
    print('Evaluation Precision on all %s images, %2.2f' % (val_type.value ,acc_val.avg.item()))
    print('Evaluation Recall on all %s images, %2.2f' % (val_type.value ,acc_val.avg.item()))
    print('Evaluation F1 Score on all %s images, %2.2f' % (val_type.value ,acc_val.avg.item()))
    print('Evaluation average AUC-ROC on all %s images, %2.2f' % (val_type.value ,acc_val.avg.item()))
    print("Evaluation Finished")
    if writer is not None:
        writer.add_text("Loss/" + val_type.value, str(loss_val.avg.item()))
        writer.add_text("Accuracy/"  + val_type.value, str(acc_val.avg.item()))
        writer.add_text("Top2Acc/"+ val_type.value, str(acc2_val.avg.item()))
        writer.add_text("Top3Acc/"+ val_type.value, str(acc3_val.avg.item()))
        writer.add_text("Precision/"+ val_type.value, str(precision))
        writer.add_text("Recall/"+ val_type.value, str(recall))
        writer.add_text("F1 Score/"+ val_type.value, str(f1_score))
        if not sys.gettrace():
            writer.add_text("Average AUC-ROC/" + val_type.value, str(roc_auc_avg))
            writer.add_figure("ROC/"+ val_type.value, roc_fig, 0)
            writer.add_figure("Confusion matrix/"+ val_type.value, conf_matrix, 0)
            plt.close('all')
        writer.close()


def evaluate(model, criterion, data_loader, val_device):
    model.eval()
    acc_avg = ut.AverageMeter('Accuracy', ':6.2f')
    loss_avg = ut.AverageMeter('Loss', ':6.2f')

    acc2_avg = ut.AverageMeter('Top 2 Accuracy', ':6.2f')
    acc3_avg = ut.AverageMeter('Top 3 Accuracy', ':6.2f')
    y_true_all = []
    y_pred_all = []
    prob_pred_all = []

    class_names = ["Triangle", "Ruler", "Rubber", "Pencil", "Pen", "None"]
    class_names_pred = [class_name + ".P" for class_name in class_names]
    classes_count = len(class_names)

    conf_matrix_heatmap = None
    roc_auc_avg = None
    roc_fig = None

    precision = 0
    recall = 0
    f1_score = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            image_dims = (data.size()[0], int(data[0][2]), int(data[0][3]), int(data[0][4]))
            image = np.reshape(data[:, 5:], image_dims)
            inputs = torch.autograd.Variable(image.to(val_device, non_blocking=True))
            labels = torch.autograd.Variable(data[:, 0].long().to(val_device, non_blocking=True))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc, acc2, acc3 = accuracy(outputs, labels, topk=(1, 2, 3))
            acc_avg.update(acc[0], inputs.size(0))
            acc2_avg.update(acc2[0], inputs.size(0))
            acc3_avg.update(acc3[0], inputs.size(0))
            loss_avg.update(loss, inputs.size(0))
            prob_pred_all += softmax(outputs, 1).tolist()
            _, pred = outputs.topk(1, 1, True, True)
            y_pred_all += pred.tolist()
            y_true_all += labels.tolist()

    if not sys.gettrace():
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        df_cm = pd.DataFrame(conf_matrix, index=[i for i in class_names],
                             columns=[i for i in class_names_pred])

        conf_matrix_heatmap = sn.heatmap(df_cm, annot=True).get_figure()
        roc_auc = roc_auc_score(y_true_all, prob_pred_all, average=None, multi_class='ovr')
        roc_auc_avg = roc_auc_score(y_true_all, prob_pred_all, average='macro', multi_class='ovr')
        y_true_binary = [[true == i for true in y_true_all] for i in range(classes_count)]
        y_pred_binary = [[probs[i] for probs in prob_pred_all]for i in range(classes_count)]
        rocs = [RocCurveDisplay.from_predictions(y_true_binary[i], y_pred_binary[i]) for i in range(classes_count)]
        rocs_x = [roc.line_.get_xdata() for roc in rocs]
        rocs_y = [roc.line_.get_ydata() for roc in rocs]
        styles = ['-r', '-g', '-b', '-c', '-m', '-y']
        roc_fig, roc_axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        roc_axs = roc_axs.flatten()
        for i in [0, 2, 8]:
            roc_axs[i].axis('off')
        for i in range(classes_count):
            roc_axs[1].plot(rocs_x[i], rocs_y[i], styles[i], label=f"{class_names[i]} - AUC={roc_auc[i]:.3f}")
            if i != 0:
                roc_ax = roc_axs[i+2]
                roc_ax.plot(rocs_x[i], rocs_y[i], styles[i], label=f"{class_names[i]} - AUC={roc_auc[i]:.3f}")
                roc_ax.legend(loc='lower right')
        roc_axs[1].legend(loc='best', fontsize='x-small')

        class_precisions = []
        class_recalls = []

        cm_sum = conf_matrix.sum()
        cm_col_sum = df_cm.sum('index')
        cm_row_sum = df_cm.sum('columns')

        for i in range(classes_count):
            tp = conf_matrix[i][i]
            fp = cm_col_sum[i] - tp
            fn = cm_row_sum[i] - tp
            weight = cm_row_sum[i] / cm_sum
            class_precision = None
            class_recall = None
            if tp != 0 or fp != 0:
                class_precision = tp / (tp + fp)
                precision += class_precision * weight
                class_precisions.append(class_precision)
            if tp != 0 or fn != 0:
                class_recall = tp / (tp + fn)
                recall += class_recall * weight
                class_recalls.append(class_recall)
            if class_precision is not None and class_recall is not None and (class_precision != 0 or class_recall != 0):
                f1_score += (2 * class_precision * class_recall / (class_precision + class_recall)) * weight

    return loss_avg, (acc_avg,acc2_avg,acc3_avg), precision, recall, f1_score, conf_matrix_heatmap, roc_auc_avg, roc_fig


def accuracy(outputs, labels , topk=(1,)):
    """Computes the accuracy over the top predictions for the specified values of k"""
    with torch.no_grad():  # disables recalculation of gradients
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


def weight_change(outputs, labels, weights, hparams):
    with torch.no_grad():
        # returns unnormalized output for prediction class, and prediction class nr
        _, predictions = outputs.topk(1,1,True,True)
        # checks if prediction is correct
        wrongs = torch.squeeze(~predictions.t().eq(labels.contiguous().view(1, -1)))
        # check against the sufit aka the max value of the weight
        delta = hparams["weights_delta"]
        ceiling = (weights + delta * wrongs) < hparams["weights_ceiling"]
        return weights + delta * ceiling * wrongs


if __name__ == '__main__':
    main()
