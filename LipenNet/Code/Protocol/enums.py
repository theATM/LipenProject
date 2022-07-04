from enum import Enum


class DatasetName(Enum):
    RawDataset: str = "raw"
    CleanDataset: str = "clean"
    UniformDataset: str = "uniform"
    UnaugDataset: str = "unaug"
    MergedDataset: str = "merged"
    MeraugDataset: str = "meraug"
    Unaug20Dataset: str = "unaug20"


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
    ConvNext: str = "ConvNext"
    ConvNextBare :str = "ConvNextBare"
    Squeezenet: str = "Squeezenet"


class AugmentationType(Enum):
    Without: str = "Without"
    Rotation: str = "Rotation"
    Online: str = "Online"
    Offline: str = "Offline"
    Normalize: str = "Normalize"
    Normrot : str = "Normrot"


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


class ValType(Enum):
    train = 'train'
    val = 'val'
    test = 'test'


labels = dict(
     triangle = 0,
     ruler    = 1,
     gum      = 2,
     pencil   = 3,
     pen      = 4,
     none     = 5,
     wrong    = 6
)

extralabels = dict(
    broken = 1,
    blurred = 2,
    dark = 4,
    lamp = 8,
    hard = 16,
    normal=0
)


