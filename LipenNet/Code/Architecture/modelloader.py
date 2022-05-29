import torch
import torchvision.models
import copy
import Code.Protocol.enums as en
from datetime import datetime
from Code.Profile.profileloader import Hparams
import os
import shutil
from typing import Optional
import Code.Architecture.model as m


def pickModel(hparams: Hparams):
    model = None
    match hparams['model']:
        case en.ModelType.Resnet18_pretrained:
            # load pretrained resnet18 model
            model = torchvision.models.resnet18(pretrained=True)
            # add fresh last layer with number of classes
            model.fc = torch.nn.Linear(model.fc.in_features, 6)  # add last layer
            layers = [model.layer1, model.layer2, model.layer3, model.layer4]
            frozen = hparams["frozen_initial_layers"]
            for layer in layers[:frozen]:
                for param in layer.parameters():
                    param.requires_grad = False

        case en.ModelType.Resnet18:
            # load resnet 18 model with fresh weights and proper class number
            model = torchvision.models.resnet18(pretrained=False, num_classes=6)
            model.fc = torch.nn.Linear(model.fc.in_features, 6)  # add last layer
            layers = [model.layer1, model.layer2, model.layer3, model.layer4]
            frozen = hparams["frozen_initial_layers"]
            for layer in layers[:frozen]:
                for param in layer.parameters():
                    param.requires_grad = False

        case en.ModelType.Alexnet:
            model = m.AlexNet(num_classes=6)
    return model


def load_model(model, optimizer, scheduler, load_device, hparams: Hparams):
    load_path = hparams['load_model_path']
    # Load File:
    model_load_dict = torch.load(load_path, map_location=load_device)
    model_states = model_load_dict["model_states"]
    optim_states = model_load_dict["optim_states"]
    schedule_states = model_load_dict["scheduler_states"]
    load_params = model_load_dict["save_params"]

    # Load Model:
    model.load_state_dict(model_states)
    optimizer.load_state_dict(optim_states)
    scheduler.load_state_dict(schedule_states)
    return load_params


def load_model_test(model, load_device, hparams: Hparams):
    load_path = hparams['load_model_path']
    # Load File:
    model_load_dict = torch.load(load_path, map_location=load_device)
    model_states = model_load_dict["model_states"]

    # Load Model:
    model.load_state_dict(model_states)


def save_model(model, optimizer, scheduler, hparams: Hparams, save_params: dict):
    # Create Savable copy of model
    model_states = copy.deepcopy(model.state_dict())
    optim_states = copy.deepcopy(optimizer.state_dict())
    scheduler_states = copy.deepcopy(scheduler.state_dict())
    # Create New Fancy Save Name
    save_name = "Model_"
    save_name += str(hparams["model"].value) + "_"
    now = datetime.now()  # Used to differentiate saved models
    now_str = now.strftime("%d.%m.%Y_%H:%M")
    save_name += str(now_str) + "_"
    if "current_epoch" in save_params:
        save_name += "Epoch_" + str(save_params["current_epoch"]) + "_"
    if "current_acc" in save_params:
        save_name += "Acc_" + str(save_params["current_acc"].item())
    save_name += ".pth"

    # Create Save Dictionary:
    model_save_dict = \
        {
            'title': "This is save file of the model of the LipenNet - the school equipment recognition network",
            'authors': "Julia Dmytrenko, Nikodem Matuszkiewicz, Aleksander Madajczak",
            'model_type': hparams["model"],
            'model_states': model_states,
            'criterion_type': hparams['criterion'],
            'optimizer_type': hparams['optimizer'],
            "optimizer_states": optim_states,
            "scheduler_states": scheduler_states,
            "save_params": save_params,
            "hyper_params": hparams,
        }

    # Save Current Model
    save_path = str(hparams['save_dir_path']) + save_name
    torch.save(model_save_dict, save_path)


def savePrepareDir(hparams: Hparams):
    # Create path
    save_dir_name = getModelName(hparams)
    hparams['save_dir_path'] += save_dir_name + "/"
    if not os.path.exists(hparams['save_dir_path'][:-1]):
        os.makedirs(hparams['save_dir_path'][:-1])
    profile_file_name = hparams["profile_file"].split("/")[-1]
    profile_file_name = profile_file_name.split(".")[0] + "_" + save_dir_name + "." + profile_file_name.split(".")[1]
    profile_file_path = hparams['save_dir_path'] + profile_file_name
    shutil.copyfile(hparams["profile_file"], profile_file_path)


def getModelName(hparams: Hparams, withdataset=False):
    if 'save_dir_name' not in hparams:
        now = datetime.now()  # Used to differentiate saved models
        now_str = now.strftime("%d.%m.%Y_%H:%M")
        if withdataset:
            dataset_name = str(hparams["dataset_name"].value) + "_"
        else:
            dataset_name = ""
        save_dir_name = str(hparams["model"].value) + "_" + dataset_name + now_str
        hparams['save_dir_name'] = save_dir_name
    return hparams['save_dir_name']


def pickCriterion(hparams: Hparams, purpose: en.CriterionPurpose = en.CriterionPurpose.EvalCriterion):
    criterion = None
    criterion_type = hparams['criterion'] if purpose == en.CriterionPurpose.TrainCriterion else hparams['val_criterion']
    reduction = hparams['reduction_mode'].value if purpose == en.CriterionPurpose.TrainCriterion else "mean"
    match criterion_type:
        case en.CriterionType.CrossEntropy:
            criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    return criterion


def pickOptimizer(model, hparams: Hparams):
    optimizer = None
    match hparams['optimizer']:
        case en.OptimizerType.Adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams['initial_learning_rate'],
                                         weight_decay=hparams['weight_decay'])
        case en.OptimizerType.AdamW:
            optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['initial_learning_rate'],
                                          weight_decay=hparams['weight_decay'])
    return optimizer

# class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
#    #https://stackoverflow.com/questions/67730325/using-weights-in-crossentropyloss-and-bceloss-pytorch
#    def __init__(self, weight: torch.Tensor):
#        super().__init__(weight,reduction='none')
#
#    def forward(self, input: torch.Tensor, target: torch.Tensor, weights:torch.Tensor) -> torch.Tensor:
#        intermediate_losses = super(torch.nn.CrossEntropyLoss, self).forward(input,target)
#        final_loss = torch.mean(weights * intermediate_losses)
