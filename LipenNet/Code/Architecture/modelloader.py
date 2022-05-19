import torch
import torchvision.models
import copy
import Code.Protocol.enums as en
from datetime import datetime
from Code.Profile.profileloader import Hparams
import os
import shutil
from typing import Optional


def pickModel(hparams:Hparams):
    model = None
    match hparams['model']:
        case en.ModelType.Resnet18_pretrained:
            model = torchvision.models.resnet18(pretrained=True,)
            model.fc.in_features = torch.nn.Linear(model.fc.in_features, 6) #add last layer

        case en.ModelType.Resnet18:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            model.fc.in_features = torch.nn.Linear(model.fc.in_features, 6)  # add last layer
    return model


def loadModel(model,optimizer,scheduler,load_device,hparams:Hparams):
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


def saveModel(model,optimizer,scheduler,hparams:Hparams,save_params:dict):
    # Create Savable copy of model
    model_states = copy.deepcopy(model.state_dict())
    optim_states = copy.deepcopy(optimizer.state_dict())
    scheduler_states = copy.deepcopy(scheduler.state_dict())
    #Create New Fancy Save Name
    save_name ="Model_"
    save_name += str(hparams["model"].value) + "_"
    now = datetime.now()  # Used to differentiate saved models
    now_str = now.strftime("%d.%m.%Y_%H:%M")
    save_name += str(now_str) + "_"
    if "current_epoch" in save_params:
        save_name += "Epoch_" + str(save_params["current_epoch"]) + "_"
    if "current_acc" in save_params:
        save_name += "Acc_" + str(save_params["current_acc"])
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
        "optimizer_states":optim_states,
        "scheduler_states":scheduler_states,
        "save_params":save_params,
        "hyper_params":hparams,
    }

    # Save Current Model
    save_path = str(hparams['save_dir_path']) + save_name
    torch.save(model_save_dict, save_path)


def savePrepareDir(hparams:Hparams):
    #Create path
    now = datetime.now()  # Used to differentiate saved models
    now_str = now.strftime("%d.%m.%Y_%H:%M")
    save_dir_name = str(hparams["model"].value) + "_" + now_str
    hparams['save_dir_path'] += save_dir_name + "/"
    os.mkdir(hparams['save_dir_path'][:-1])
    profile_file_name = hparams["profile_file"].split("/")[-1]
    profile_file_name = profile_file_name.split(".")[0] +"_" + save_dir_name +"."+ profile_file_name.split(".")[1]
    profile_file_path = hparams['save_dir_path'] + profile_file_name
    shutil.copyfile(hparams["profile_file"],profile_file_path)






def pickCriterion(hparams:Hparams):
    criterion = None
    match hparams['criterion']:
        case en.CriterionType.CrossEntropy:
            criterion = torch.nn.CrossEntropyLoss()
        case en.CriterionType.WeightedCrossEntropyLoss:
            criterion = WeightedCrossEntropyLoss(weight=torch.FloatTensor([1.0,1.0,1.0,1.0,1.0,1.0]))
    return criterion


def pickOptimizer(model,hparams:Hparams):
    optimizer = None
    match hparams['optimizer']:
        case en.OptimizerType.Adam:
            optimizer = torch.optim.Adam(model.parameters(),lr=hparams['initial_learning_rate'])
    return optimizer


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    #https://stackoverflow.com/questions/67730325/using-weights-in-crossentropyloss-and-bceloss-pytorch
    def __init__(self, weight: torch.Tensor):
        super().__init__(weight,reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights:torch.Tensor) -> torch.Tensor:
        intermediate_losses = super(torch.nn.CrossEntropyLoss, self).forward(input,target)
        final_loss = torch.mean(weights * intermediate_losses)

