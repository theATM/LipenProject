from enum import Enum


class DatasetName(Enum):
    RawDataset: str = "raw"
    CleanDataset: str = "clean"
    UniformDataset: str = "unified"
    UnaugDataset: str = "unaug"
    MergedDataset: str = "merged"
    MeraugDataset: str = "meraug"
    Uniform20Dataset: str = "uniform20"


class DatasetType(Enum):
    Trainset :str = "Trainset"
    Testset: str = "Testset"
    ValSet: str = "ValSet"


class CriterionType(Enum):
    CrossEntropy : str = "CrossEntropy"
    WeightedCrossEntropy : str = "WeightedCrossEntropy"


class CriterionPurpose(Enum):
    TrainCriterion  = 0
    EvalCriterion = 1


class OptimizerType(Enum):
    Adam :str = "Adam"
    AdamW: str = "AdamW"


class ModelType(Enum):
    Resnet18_pretrained :str = "Resnet18_pretrained"
    Resnet18 : str = "Resnet18"
    Alexnet : str = "Alexnet"


class AugmentationType(Enum):
    Without: str = "Without"
    Rotation: str = "Rotation"
    Online: str = "Online"
    Offline: str = "Offline"
    Normalize: str = "Normalize"


class Device(Enum):
    Cuda :str = "cuda"
    Cpu :str = "cpu"


class SavingMode(Enum):
    all_save = "all_save"
    best_save = "best_save"
    last_save = "last_save"
    none_save = "mone_save"


class ReductionMode(Enum):
    none = 'none'
    mean = "mean"
    sum = "sum"




