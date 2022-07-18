import functools
import typing as th
import torch
import torchvision
import pytorch_lightning as pl
from torchde.utils import FunctionDescriptor, process_function_description, safe_function_call_wrapper


class CheckOutlierCallback(pl.Callback):
    """Evaluate the model by recording the nll assigned to random noise.

    If the training loss stays constant across iterations, this score is likely to show
     the progress of the model in detecting "outliers".

    Attributes:
        batch_size (int): Number of random samples to check
        every_n_epochs (int): Check every n epochs
    """

    def __init__(
        self,
        inputs_shape: th.Union[tuple, list],
        name: th.Optional[str] = None,
        inputs_range: th.Union[tuple, list] = (-1.0, 1.0),
        criterion: th.Optional[FunctionDescriptor] = None,
        criterion_roi: th.Optional[th.Union[th.List[str], str]] = None,
        batch_size: int = 32,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.name = f"{name}/" if name is not None else ""
        self.batch_size = batch_size
        self.inputs_shape = tuple(inputs_shape)
        self.every_n_epochs = every_n_epochs
        self.criterion_description = criterion
        self.criterion_roi = criterion_roi
        self.inputs_range = inputs_range

    @functools.cached_property
    def criterion_function(self):
        if self.criterion_description is None:
            return None
        return safe_function_call_wrapper(
            process_function_description(self.criterion_description, entry_function="criterion")
        )

    def criterion(self, inputs: torch.Tensor, trainer: pl.Trainer, pl_module: pl.LightningModule) -> torch.Tensor:
        if self.criterion_function is None:
            return pl_module.criterion(training_module=pl_module, model=pl_module.model, inputs=inputs)
        return self.criterion_function(inputs=inputs, training_module=pl_module, trainer=trainer)

    def compute(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            rand_inputs = torch.rand(self.batch_size, *self.inputs_shape, device=pl_module.device)
            rand_inputs = rand_inputs * (self.inputs_range[1] - self.inputs_range[0]) + self.inputs_range[0]
            results = self.criterion(pl_module=pl_module, trainer=trainer, inputs=rand_inputs)
        if self.criterion_roi is None:
            trainer.logger.experiment.add_scalar(
                f"metrics/outliers/{self.name}score", results[self.criterion_roi], global_step=trainer.global_step
            )
        elif isinstance(self.criterion_roi, str):
            trainer.logger.experiment.add_scalar(
                f"metrics/outliers/{self.name}{self.criterion_roi}",
                results[self.criterion_roi],
                global_step=trainer.global_step,
            )
        else:
            for roi in self.criterion_roi:
                trainer.logger.experiment.add_scalar(
                    f"metrics/outliers/{self.name}{roi}", results[roi], global_step=trainer.global_step
                )

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch % self.every_n_epochs:
            return
        return self.compute(trainer=trainer, pl_module=pl_module)


class SampleAdversariesCallback(pl.Callback):
    """Evaluate the model by sampling adversarial examples.

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
        log_original_inputs: bool = True,
        original_grid_args: th.Optional[dict] = None,
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
        self.original_grid_args = {**(grid_args or dict()), **(original_grid_args or dict())}
        self.adversaries_grid_args = {**(grid_args or dict()), **(adversaries_grid_args or dict())}
        self.difference_map_grid_args = {**(grid_args or dict()), **(difference_map_grid_args or dict())}
        self.difference_map_normalize = difference_map_normalize
        self.log_original_inputs = log_original_inputs

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
                inputs, labels = pl_module.process_inputs(batch)
                yield inputs.to(pl_module.device)

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
        return pl_module.attacker(inputs=inputs, training_module=pl_module, return_loss=False)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (
            trainer.current_epoch % self.every_n_epochs  # interval
            or not hasattr(pl_module, "attacker")  # no attacker for this model
            or pl_module.attacker is None  # attacker is not set
        ):
            return

        inputs = next(self.inputs_iterator)
        adversaries = self.get_adversaries(trainer=trainer, pl_module=pl_module, inputs=inputs)

        # log original inputs
        if self.log_original_inputs:
            trainer.logger.experiment.add_image(
                f"attack/original_inputs",
                torchvision.utils.make_grid(
                    self.process_shape(inputs, force_grayscale=False), **self.adversaries_grid_args
                ),
                global_step=trainer.global_step,
            )
        # visualize adversaries
        trainer.logger.experiment.add_image(
            f"attack/adversaries",
            torchvision.utils.make_grid(
                self.process_shape(adversaries, force_grayscale=False), **self.adversaries_grid_args
            ),
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
