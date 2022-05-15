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
    A :str = "A"  #TODO


class AugmentationType(Enum):
    Without :str = "Without"
    Rotation :str = "Rotation"
    Online :str = "Online"
    Normalize :str = "Normalize"


class Device(Enum):
    Cuda :str = "cuda"
    Cpu :str = "cpu"

