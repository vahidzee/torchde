import torch

try:
    from pytorch_lightning.utilities.cli import LightningCLI
except ImportError:
    import pip

    pip.main(["install", "-U", "pytorch-lightning[extra]"])
    from pytorch_lightning.utilities.cli import LightningCLI

from made.training import MADETrainer
from made.data import DEDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args((torch.optim.SGD, torch.optim.Adam))
        parser.add_lr_scheduler_args(
            (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler._LRScheduler)
        )


if __name__ == "__main__":
    MyLightningCLI(MADETrainer, DEDataModule)
