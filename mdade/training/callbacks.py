import typing as th
import torch
import torchvision
import pytorch_lightning as pl


class CheckOutlierCallback(pl.Callback):
    """Evaluate the model by recording the nll assigned to random noise.

    If the training loss stays constant across iterations, this score is likely to show
     the progress of the model in detecting "outliers".

    Attributes:
        batch_size (int): Number of random samples to check
        every_n_epochs (int): Check every n epochs
    """

    def __init__(self, batch_size: int = 32, every_n_epochs: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs

    def compute(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            pl_module.eval()
            # todo: support non-mlp based models
            rand_inputs = torch.rand(self.batch_size, pl_module.model.in_features, device=pl_module.device)
            rand_out = pl_module.criterion(model=pl_module.model, inputs=rand_inputs)
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_nll", rand_out, global_step=trainer.global_step)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        return self.compute(trainer=trainer, pl_module=pl_module)


class SampleAdversariesCallback(pl.Callback):
    """Evaluate the model by sampling adversarial examples.

    If the training loss stays constant across iterations, this score is likely to show
     the progress of the model in detecting "outliers".

    Attributes:
        dataloader (str): dataloader (val/train/test) to use for initial inputs
        dataset_args (dict): arguments to pass to the dataloader (if to be different than the original)
        reshape (tuple): reshape the inputs to this shape [H, W]
        difference_map_overlay_alpha (float):
            alpha value for the difference map overlay (0 not to log difference maps)
        difference_map_normalize (bool): normalize the difference map values to [-1, 1]
        adversaries_grid_args (dict): arguments to pass to the grid visualization of the adversaries
        difference_map_grid_args (dict): arguments to pass to the grid visualization of the difference maps
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        dataloader="train",
        dataloader_args: th.Optional[dict] = None,
        reshape: th.Optional[th.Union[list, th.Tuple[int, int]]] = None,
        difference_map_overlay_alpha: float = 0.6,
        difference_map_normalize: bool = False,
        grid_args: th.Optional[dict] = None,
        adversaries_grid_args: th.Optional[dict] = None,
        difference_map_grid_args: th.Optional[dict] = None,
    ):
        super().__init__()

        # interval settings
        self.every_n_epochs = every_n_epochs

        # dataloader attributes
        self.dataloader_name = dataloader
        self.dataloder = None  # will be added in setup stage
        self.dataloader_args = dataloader_args or dict()

        # visualization configs
        self.reshape = tuple(reshape) if reshape else None
        self.difference_map_overlay_alpha = difference_map_overlay_alpha
        self.adversaries_grid_args = {**(grid_args or dict()), **(adversaries_grid_args or dict())}
        self.difference_map_grid_args = {**(grid_args or dict()), **(difference_map_grid_args or dict())}
        self.difference_map_normalize = difference_map_normalize

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: th.Optional[str] = None) -> None:
        if stage == "fit":
            self.dataloader = getattr(trainer.datamodule, f"{self.dataloader_name}_dataloader")(**self.dataloader_args)
            self.inputs_iterator = iter(self.get_inputs_generator(trainer=trainer, pl_module=pl_module))

    def process_shape(self, inputs, force_grayscale=True, three_channel_grayscales=True):
        """Process the shape of the inputs to be used in the visualization.

        If the input tensors are reshaped image tensors, they shall be reshaped to the provided width and height.
        Else, if the input tensors represent non-single channel images, they shall be converted to grayscale.
        """
        results = inputs.reshape(inputs.shape[0], -1, *self.reshape) if self.reshape else inputs
        if results.shape[1] == 1:
            results = results.repeat(1, 3 if three_channel_grayscales else 1, 1, 1)
        elif force_grayscale or (results.shape[1] > 3 or results.shape[1] == 2):
            results = results.mean(1, keepdim=True)  # convert to grayscale
            results = results.repeat(1, 3 if three_channel_grayscales else 1, 1, 1)
        return results

    def process_difference_map(
        self, adversarial_inputs, inputs, overlay_alpha: th.Optional[float] = None, normalize: th.Optional[bool] = None
    ):
        # config
        overlay_alpha = overlay_alpha if overlay_alpha is not None else self.difference_map_overlay_alpha
        normalize = normalize if normalize is not None else self.difference_map_normalize

        # process shapes
        difference_map = self.process_shape(adversarial_inputs - inputs, three_channel_grayscales=False)
        inputs = self.process_shape(inputs)
        # compute difference maps (positive changes are red, negative changes are blue)
        changes = torch.zeros_like(inputs)
        changes[:, 0] = torch.where(difference_map > 0, difference_map, 0)[:, 0]  # red
        changes[:, 2] = torch.where(difference_map < 0, -difference_map, 0)[:, 0]  # blue
        if normalize:
            changes /= changes.abs().max()
        return inputs * (1 - overlay_alpha) + overlay_alpha * changes

    def get_inputs_generator(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> torch.Tensor:
        """Retrieve and process a batch of inputs from the dataloader.

        Returns:
            Tensor of inputs
        """
        while True:
            for batch in self.dataloader:
                results = pl_module.process_inputs(batch).to(pl_module.device)
                yield results

    def get_adversaries(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Get adversaries to the model

        Arguments:
            trainer: trainer
            pl_module: pl_module
            inputs: inputs to the model

        Returns:
            Tensor of adversarial inputs
        """
        return pl_module.attacker(inputs=inputs, model=pl_module.model, return_loss=False)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (
            trainer.current_epoch % self.every_n_epochs  # interval
            or not hasattr(pl_module, "attacker")  # no attacker for this model
            or pl_module.attacker is None  # attacker is not set
        ):
            return

        inputs = next(self.inputs_iterator)
        adversaries = self.get_adversaries(trainer=trainer, pl_module=pl_module, inputs=inputs)

        # visualize adversaries
        trainer.logger.experiment.add_image(
            f"attack/adversaries",
            torchvision.utils.make_grid(self.process_shape(adversaries), **self.adversaries_grid_args),
            global_step=trainer.global_step,
        )

        # visualize difference maps
        if self.difference_map_overlay_alpha:
            pl_module.logger.experiment.add_image(
                "attack/difference_map",
                torchvision.utils.make_grid(
                    self.process_difference_map(adversarial_inputs=adversaries, inputs=inputs),
                    **self.difference_map_grid_args,
                ),
                global_step=trainer.global_step,
            )


class SampleCallback(pl.Callback):
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
        every_n_epochs: int = 5,
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
