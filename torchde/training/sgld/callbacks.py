import pytorch_lightning as pl
import torch
import random
import torchvision
import typing as th
import functools
from torchde.utils import FunctionDescriptor, process_function_description, safe_function_call_wrapper
from .sampler import SGLDSampler


class SGLDLogSamplerBufferCallback(pl.Callback):
    """Log a randomly picked subset of images in the SGLDSampler's buffer.

    Attributes:
        num_samples (int): number of buffer samples picked (default: 32)
        every_n_epochs (int): log interval (default: 5)
        grid_args (dict): arguments to pass to make_grid (default: {})
    """

    def __init__(
        self,
        num_samples: int = 32,
        every_n_epochs: int = 5,
        grid_args: th.Optional[dict] = None,
    ):
        super().__init__()
        self.num_samples = num_samples  # number of images to plot
        self.every_n_epochs = every_n_epochs
        self.grid_args = grid_args or {}

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Automatically gets called upon the ending of each training epoch

        Args:
            trainer: pytorch_lightning trainer module
            pl_module: module being trained
        Returns:
            None
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choices(pl_module.sampler.buffer, k=self.num_samples), dim=0)
            grid = torchvision.utils.make_grid(exmp_imgs, **self.grid_args)
            trainer.logger.experiment.add_image("SGLDSampler/buffer", grid, global_step=trainer.global_step)


class SGLDLogSamplesCallback(pl.Callback):
    def __init__(
        self,
        # callback settings
        name: th.Optional[str] = None,
        every_n_epochs: int = 5,
        num_samples: int = 8,
        visualize_steps: th.Optional[th.Union[bool, int]] = False,
        grid_args: th.Optional[dict] = None,
        # sample generation configuarations
        num_steps: th.Optional[int] = None,
        step_size: th.Optional[int] = None,
        noise_eps: th.Optional[int] = None,
        grad_clamp: th.Optional[th.Union[tuple, list]] = None,
        inputs_value_range: th.Optional[th.Union[tuple, list]] = None,
        inputs_shape: th.Optional[th.Union[tuple, list]] = None,
        energy_function: th.Optional[FunctionDescriptor] = None,
        buffer_replay_prob: th.Optional[float] = None,
    ):
        super().__init__()
        # callback settings
        self.name = f"{name}/" if name else ""
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.visualize_steps = visualize_steps
        self.grid_args = grid_args or {}
        self.energy_function_descriptor = energy_function
        self.num_steps = num_steps
        self.step_size = step_size
        self.noise_eps = noise_eps
        self.grad_clamp = grad_clamp
        self.buffer_replay_prob = buffer_replay_prob
        self.inputs_shape = tuple(inputs_shape) if inputs_shape is not None else None
        self.inputs_value_range = tuple(inputs_value_range) if inputs_value_range else None

    @functools.cached_property
    def energy_function(self):
        if self.energy_function_descriptor is None:
            return None
        return safe_function_call_wrapper(
            process_function_description(self.energy_function_descriptor, entry_function="energy")
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs:
            return
        samples = self.generate_imgs(pl_module)
        if not self.visualize_steps:
            grid = torchvision.utils.make_grid(samples, **self.grid_args)
            trainer.logger.experiment.add_image(
                f"SGLDSampler/{self.name}samples", grid, global_step=trainer.global_step
            )
            return
        grid = torchvision.utils.make_grid(
            samples.reshape(-1, *samples.shape[2:]),**{"nrow": samples.shape[0], **self.grid_args}
        )
        trainer.logger.experiment.add_image(
            f"SGLDSampler/{self.name}generation", grid, global_step=trainer.global_step
        )

    def generate_imgs(self, pl_module):
        if hasattr(pl_module, "sampler"):
            samples = pl_module.sampler.sample(
                sample_size=self.num_samples,
                num_steps=self.num_steps,
                step_size=self.step_size,
                noise_eps=self.noise_eps,
                grad_clamp=self.grad_clamp,
                buffer_replay_prob=self.buffer_replay_prob,
                return_samples_per_step=self.visualize_steps,
                inputs_value_range=self.inputs_value_range,
                energy_function=self.energy_function,
                update_buffer=False,
                device=pl_module.device,
            )
        else:
            samples = SGLDSampler.generate_samples(
                model=pl_module,
                init_inputs=SGLDSampler.generate_rand_inputs(
                    self.num_samples, self.inputs_shape, self.inputs_value_range
                ),
                sample_size=self.num_samples,
                num_steps=self.num_steps,
                step_size=self.step_size,
                noise_eps=self.noise_eps,
                grad_clamp=self.grad_clamp,
                buffer_replay_prob=self.buffer_replay_prob,
                return_samples_per_step=self.visualize_steps,
                inputs_value_range=self.inputs_value_range,
                energy_function=self.energy_function,
                training_module=pl_module,
                device=pl_module.device,
            )
        return samples
