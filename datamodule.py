import lightning as L
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


class DigitDataModule(L.LightningDataModule):
    def __init__(self, dict_size: int, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        # Setup the transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x[0] * (dict_size - 2) + 1).long()),
                # we fill the padding with 1 since 0 is the mask token
                transforms.Pad((2, 2, 2, 2), fill=1, padding_mode="constant"),
            ]
        )

    def prepare_data(self):
        MNIST("MNIST", train=True, download=True)
        MNIST("MNIST", train=False, download=True)

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            full_set = MNIST(
                root="MNIST",
                train=True,
                transform=self.transform,
                download=True,
            )
            train_set_size = int(len(full_set) * 0.8)
            val_set_size = len(full_set) - train_set_size
            seed = torch.Generator().manual_seed(42)
            (
                self.train_set,
                self.val_set,
            ) = data.random_split(  # Split train/val datasets
                full_set, [train_set_size, val_set_size], generator=seed
            )
        elif stage == "test":
            self.test_set = MNIST(
                root="MNIST",
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=10
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=10
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=10
        )