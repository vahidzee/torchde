from pytorch_lightning.utilities.cli import LightningCLI
from torchde.training import DETrainingModule
from torchde.data import DEDataModule
from torchde.__about__ import __version__


def main():
    print(f"torchde version {__version__} invoked")
    LightningCLI(
        DETrainingModule,
        DEDataModule,
        subclass_mode_model=True,
    )


if __name__ == "__main__":
    main()
