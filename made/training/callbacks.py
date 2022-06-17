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
    """This callback evaluates the model by recording the (negative) energy assigned to random noise. While our training
    loss is almost constant across iterations, this score is likely showing the progress of the model to detect
    "outliers".

    Attributes:
        batch_size (int): Number of random samples to check
    """

    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def log(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            pl_module.eval()
            # todo: support non-mlp based models
            rand_inputs = torch.rand(
                self.batch_size, pl_module.model.in_features, device=pl_module.device
            )
            rand_out = pl_module.criterion(model=pl_module.model, inputs=rand_inputs)
            pl_module.train()

        trainer.logger.experiment.add_scalar(
            "rand_nll", rand_out, global_step=trainer.global_step
        )

    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return self.log(trainer=trainer, pl_module=pl_module)


class SampleAdversariesCallback(pl.Callback):
    pass


class SampleCallback(pl.Callback):
    """
    Attributes:
        num_samples (int): Number of images to generate
        every_n_epochs (int): Only save those images every N epochs (otherwise logs gets quite large)
        reshape: optionally reshape model samples (for mlp-based models)
    """

    def __init__(
        self,
        num_samples: int = 8,
        every_n_epochs: int = 5,
        reshape: th.Optional[th.Union[tuple, list]] = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.reshape = reshape

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.every_n_epochs == 0:
            samples = pl_module.model.sample(num_samples=self.num_samples)
            if self.reshape is not None:
                samples = samples.reshape(samples.shape[0], *self.reshape)
            trainer.logger.experiment.add_image(
                f"sample",
                torchvision.utils.make_grid(samples),
                global_step=trainer.global_step,
            )
