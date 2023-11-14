import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .custom_dataset import CustomDataset


class CustomDataModule(pl.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.root = config["data"].get("root", None)
        self.train_val_split_ratio = config["data"].get(
            "train_val_split_ratio", 0.8)
        self.batch_size = config["data"].get("batch_size", 4)
        self.shuffle = config["data"].get("shuffle", True)

        assert self.root is not None, "Please specify the root directory of the data set"
        assert isinstance(
            self.train_val_split_ratio, float
        ) and self.train_val_split_ratio > 0 and self.train_val_split_ratio < 1, "Please specify a valid train/val split ratio"
        assert isinstance(
            self.batch_size,
            int) and self.batch_size > 0, "Please specify a valid batch size"
        assert isinstance(
            self.shuffle,
            bool), "Please specify a valid boolean value for shuffle"

        # Separate transforms for training and validation
        self.train_transforms = transforms.Compose([
            getattr(transforms, t["name"])(**t["params"])
            for t in config["data"].get("train_transforms", [])
        ])
        self.val_transforms = transforms.Compose([
            getattr(transforms, t["name"])(**t["params"])
            for t in config["data"].get("val_transforms", [])
        ])

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(root=self.root)
        train_size = int(self.train_val_split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset_subset, self.val_dataset_subset = random_split(
            full_dataset, [train_size, val_size])
        self.train_dataset = CustomDataset(self.train_dataset_subset,
                                           transform=self.train_transforms)
        self.val_dataset = CustomDataset(self.val_dataset_subset,
                                         transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
