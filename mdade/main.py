import torch
from pytorch_lightning.utilities.cli import LightningCLI
from mdade.training import MADETrainer
from mdade.data import DEDataModule



class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args((torch.optim.SGD, torch.optim.Adam))
        parser.add_lr_scheduler_args((torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler._LRScheduler))



def main():
    MyLightningCLI(MADETrainer, DEDataModule)


if __name__ == "__main__":
    main()
