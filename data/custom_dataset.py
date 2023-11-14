from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CustomDataset(Dataset):

    def __init__(self,
                 dataset: torch.utils.data.dataset.Subset,
                 transform: Optional[Callable] = None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
