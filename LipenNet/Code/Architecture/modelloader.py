import torch
import Code.Protocol.enums as en
from Code.Profile.profileloader import Hparams


def pickModel(hparams:Hparams):
    model = None
    match hparams['train_model']:
        case en.ModelType.A:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        case en.ModelType.B:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

    return model


def pickCriterion(hparams:Hparams):
    criterion = None
    match hparams['train_criterion']:
        case en.CriterionType.CrossEntropy:
            criterion = torch.nn.CrossEntropyLoss()
    return criterion


def pickOptimizer(model,hparams:Hparams):
    optimizer = None
    match hparams['train_optimizer']:
        case en.OptimizerType.Adam:
            criterion = torch.optim.Adam(model.parameters(),lr=hparams['train_initial_learning_rate'])
    return optimizer
