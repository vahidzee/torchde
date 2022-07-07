import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torchde.training import MADETrainer
from torchde.data import DEDataModule
from torchde.__about__ import __version__


def main():
    print(f"torchde version {__version__} invoked")
    LightningCLI(MADETrainer, DEDataModule)


if __name__ == "__main__":
    main()
