import typing as th
import torch
import torchvision

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    # install pytorch_lightning if it wasn't installed
    import pip

    pip.main(["install", "pytorch_lightning>=1.6"])
    import pytorch_lightning as pl


class CheckOutlierCallback(pl.Callback):
    """Evaluate the model by recording the nll assigned to random noise.

    If the training loss stays constant across iterations, this score is likely to show
     the progress of the model in detecting "outliers".

    Attributes:
        batch_size (int): Number of random samples to check
    """

    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def compute(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            pl_module.eval()
            # todo: support non-mlp based models
            rand_inputs = torch.rand(self.batch_size, pl_module.model.in_features, device=pl_module.device)
            rand_out = pl_module.criterion(model=pl_module.model, inputs=rand_inputs)
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_nll", rand_out, global_step=trainer.global_step)

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return self.compute(trainer=trainer, pl_module=pl_module)


class SampleAdversariesCallback(pl.Callback):
    """Evaluate the model by sampling adversarial examples.

    If the training loss stays constant across iterations, this score is likely to show
     the progress of the model in detecting "outliers".

    Attributes:
        batch_size (int): Number of random samples to check
        dataset (str): dataloader (val/train/test) to use for initial inputs
    """

    def __init__(self, batch_size=16, dataloader="train"):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: th.Optional[str] = None) -> None:
        pass

    def compute(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.compute(trainer=trainer, pl_module=pl_module)


class SampleCallback(pl.Callback):
    """
    Samples generations from the model and save them to the experiment log.

    Attributes:
        num_samples (int): Number of images to generate
        every_n_epochs (int): Only save those images every N epochs (otherwise logs gets quite large)
        reshape: optionally reshape model samples (for mlp-based models)
        grid_args: arguments to pass to torchvision.utils.make_grid
    """

    def __init__(
        self,
        num_samples: int = 8,
        every_n_epochs: int = 5,
        mask_index: th.Optional[int] = None,
        reshape: th.Optional[th.Union[tuple, list]] = None,
        grid_args: th.Optional[dict] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.reshape = reshape
        self.grid_args = grid_args or dict()

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Samples generations from the model and save them to the experiment log."""
        if trainer.current_epoch % self.every_n_epochs == 0:
            samples = pl_module.model.sample(num_samples=self.num_samples)
            if self.reshape is not None:
                samples = samples.reshape(samples.shape[0], *self.reshape)
            trainer.logger.experiment.add_image(
                f"sample",
                torchvision.utils.make_grid(samples, **self.grid_args),
                global_step=trainer.global_step,
            )
