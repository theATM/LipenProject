from enum import Enum


class DatasetName(Enum):
    RawDataset: str = "raw"
    CleanDataset: str = "clean"
    UniformDataset: str = "unified"
    MergedDataset: str = "merged"


class DatasetType(Enum):
    Trainset :str = "Trainset"
    Testset: str = "Testset"
    ValSet: str = "ValSet"


class CriterionType(Enum):
    CrossEntropy : str = "CrossEntropy"


class OptimizerType(Enum):
    Adam :str = "Adam"


class ModelType(Enum):
    Resnet18_pretrained :str = "Resnet18_pretrained"
    Resnet18 : str = "Resnet18"


class AugmentationType(Enum):
    Without :str = "Without"
    Rotation :str = "Rotation"
    Online :str = "Online"
    Normalize :str = "Normalize"


class Device(Enum):
    Cuda :str = "cuda"
    Cpu :str = "cpu"

