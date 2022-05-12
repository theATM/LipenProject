from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Lipenset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.images = []


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass