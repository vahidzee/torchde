import torch
from pytorch_lightning.utilities.cli import LightningCLI
from mdade.training import MADETrainer
from mdade.data import DEDataModule
from mdade.__about__ import __version__


def main():
    print(f"mdade version {__version__} invoked")
    LightningCLI(MADETrainer, DEDataModule)


if __name__ == "__main__":
    main()
