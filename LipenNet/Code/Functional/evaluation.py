import torch
import sys

import Code.Functional.utilities as u
import Code.Dataloader.lipenset as dl
import Code.Profile.profileloader as pl


def main():
    hparams: pl.Hparams = pl.loadProfile(sys.argv)
    val_device = torch.device(hparams['val_device'].value) #TODO
    # Empty GPU Cache before Training starts
    if val_device == 'cuda': torch.cuda.empty_cache()
    # Load Data
    _ , _,testloader  = dl.loadData(hparams)
    model = ml.pickModel(hparams)

    #Load Model
    used_model = mod.UsedModel(par.MODEL_USED_MODEL_TYPE, arg_load_path=par.MODEL_LOAD_MODEL_PATH, arg_load=True,
                               arg_load_device=eval_device,  arg_load_quantized=par.EVAL_LOAD_MODEL_IS_QUANTIZED)
    used_model.model.to(eval_device)

    #Eval
    print("\nEvaluation Started")
    printdatasetName()
    used_model.model.eval()
    model_accuracy, _,_ = evaluate(used_model, testloader, eval_device)
    print('Evaluation Accuracy on all test images, %2.2f' % (model_accuracy.avg))
    print("Evaluation Finished")



def evaluate(model,criterion, data_loader, hparams: Hparams):
    val_device = None #TODO
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = torch.autograd.Variable(data['image'].to(val_device, non_blocking=True))
            labels = torch.autograd.Variable(data['label'].to(val_device, non_blocking=True))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #TODO accuracy
            acc = accuracy(outputs,labels)

    return loss


def accuracy(outputs, labels):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad(): # disables recalculation of gradients
        maxk = max(1)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.contiguous().view(1, -1).expand_as(pred))

        res = []

        correct = correct[:1].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
