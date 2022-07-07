import torchvision
import pytorch_lightning as pl
import typing as th


class MADESampleCallback(pl.Callback):
    """
    Samples generations from the model and save them to the experiment log.

    Attributes:
        num_samples (int): Number of images to generate
        every_n_epochs (int): Only save those images every N epochs (otherwise logs gets quite large)
        reshape: optionally reshape model samples (for mlp-based models)
        grid_args: arguments to pass to torchvision.utils.make_grid
        mask_index: index of the mask to use for sampling from model (-1 for all masks)
    """

    def __init__(
        self,
        num_samples: int = 8,
        every_n_epochs: int = 1,
        reshape: th.Optional[th.Union[th.Union[int, int], list]] = None,
        grid_args: th.Optional[dict] = None,
        mask_index: th.Optional[int] = 0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.reshape = reshape
        self.mask_index = mask_index
        self.grid_args = grid_args or dict()

    def log_samples(self, trainer: pl.Trainer, name: str, samples):
        trainer.logger.experiment.add_image(
            name,
            torchvision.utils.make_grid(samples, **self.grid_args),
            global_step=trainer.global_step,
        )

    def get_samples(self, pl_module: pl.LightningModule, mask_index: th.Optional[int] = None):
        samples = pl_module.model.sample(num_samples=self.num_samples, mask_index=mask_index)
        if self.reshape is not None:
            samples = samples.reshape(samples.shape[0], *self.reshape)
        return samples

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Samples generations from the model and save them to the experiment log."""
        if trainer.current_epoch % self.every_n_epochs:  # interval check
            return

        if self.mask_index == -1 and pl_module.model.num_masks > 1:
            # sample all masks
            for i in range(pl_module.model.num_masks):
                self.log_samples(
                    trainer=trainer,
                    name=f"samples/mask/{i}",
                    samples=self.get_samples(pl_module=pl_module, mask_index=i),
                )
            return
        self.log_samples(
            trainer=trainer,
            name=f"samples/mask/{self.mask_index}",
            samples=self.get_samples(pl_module=pl_module, mask_index=self.mask_index),
        )
