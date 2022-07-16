# codes adapted from UVA's deep learning notebooks: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
import torch
import random
import numpy as np
import typing as th
import functools
import torchde.utils
import torchde.training.utils


class SGLDSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        inputs_shape: th.Union[tuple, list],
        inputs_value_range: th.Optional[th.Union[tuple, list]] = (-1.0, 1.0),
        # energy
        energy_function: th.Optional[torchde.utils.FunctionDescriptor] = None,
        # buffer
        buffer_replay_prob: float = 0.95,
        buffer_size: int = 8192,
        # algorithm
        num_steps: int = 64,
        step_size: float = 10.0,
        noise_eps: th.Optional[float] = 0.005,
        grad_clamp: th.Optional[th.Union[tuple, float]] = (-0.03, 0.03),
    ):
        """
        Args:
            model: Neural network to use for modeling E_theta
            inputs_shape (tuple, list): Shape of the inputs to the model
            inputs_value_range (tuple, list, optional): Range of the inputs to the model (defaults to (-1.0, 1.0))
            buffer_replay_prob (float): Probability of taking initial samples from sample buffer (defaults to 0.95)
            buffer_size (int): Size of the sample buffer (defaults to 8192)
            num_steps (int): Number of iterations in the MCMC algorithm (defaults to 64)
            step_size (float): Learning rate nu in the SGLD's algorithm (defaults to 10.0)
            noise_eps (float, optional): Noise std to add to initial sample inputs (defaults to 0.005)
            grad_clamp (tuple, int, optional): Clamp bounds for step gradients (defaults to (-0.03, 0.03))
        """
        super().__init__()
        self.model = model
        self.inputs_shape = tuple(inputs_shape)
        self.inputs_value_range = None if inputs_value_range is None else tuple(inputs_value_range)
        # energy
        self.energy_function_descriptor = energy_function

        # buffer
        assert 0.0 <= buffer_replay_prob <= 1.0, "buffer_replay_prob should be a probability (between zero and one)"
        self.buffer_size, self.buffer_replay_prob = buffer_size, buffer_replay_prob
        self.buffer = [(torch.rand((1,) + inputs_shape) * 2 - 1) for _ in range(self.buffer_size)]
        # algorithm
        self.num_steps, self.step_size = num_steps, step_size
        self.noise_eps = noise_eps
        self.grad_clamp = grad_clamp

    @functools.cached_property
    def energy_function(self):
        if self.energy_function_descriptor is None:
            return None
        return torchde.utils.safe_function_call_wrapper(
            torchde.utils.process_function_description(self.energy_function_descriptor, entry_function="energy")
        )

    @staticmethod
    def generate_rand_inputs(
        num_samples: int,
        inputs_shape: th.Optional[th.Union[tuple, list]] = None,
        inputs_value_range: th.Optional[th.Union[tuple, list]] = None,
    ) -> torch.Tensor:
        uniform_samples = torch.rand((num_samples,) + inputs_shape)
        if inputs_value_range is None:
            return uniform_samples
        return uniform_samples * (inputs_value_range[1] - inputs_value_range[0]) + inputs_value_range[0]

    def sample(
        self,
        sample_size: int,
        num_steps: th.Optional[int] = None,
        step_size: th.Optional[float] = None,
        noise_eps: th.Optional[float] = None,
        grad_clamp: th.Optional[th.Union[tuple, list]] = None,
        energy_function: th.Optional[torchde.utils.FunctionDescriptor] = None,
        inputs_value_range: th.Optional[th.Union[tuple, list]] = None,
        buffer_replay_prob: th.Optional[float] = None,
        update_buffer: bool = True,
        return_samples_per_step: th.Union[int, bool] = False,
        device: str = "cpu",
    ):
        """Get a new batch of "fake" inputs.

        Args:
            sample_size (int): Number of samples to return
            num_steps (int, optional): Number of iterations in the MCMC algorithm (defaults to self.num_steps)
            step_size (float, optional): Learning rate nu in the algorithm above (defaults to self.step_size)
            noise_eps (float, optional): Noise std to add to initial sample inputs (defaults to self.noise_eps)
            grad_clamp (tuple, int, optional): Clamp bounds for step gradients (defaults to self.grad_clamp)
            energy_function (torchde.utils.FunctionDescriptor, optional): Energy function to use for the algorithm (defaults to self.energy_function)
            inputs_value_range (tuple, list, optional): Range of the inputs to the model (defaults to self.inputs_value_range)
            buffer_replay_prob (float, optional): Probability of taking initial samples from sample buffer (defaults to self.buffer_replay_prob)
            update_buffer (bool): Whether to update sample buffer (defaults to True)
            device (str): Which device bears the sampling computation
        """
        num_steps = num_steps if num_steps is not None else self.num_steps
        step_size = step_size if step_size is not None else self.step_size
        buffer_replay_prob = buffer_replay_prob if buffer_replay_prob is not None else self.buffer_replay_prob

        # choose buffer_replay_prob * 100% of the batch from the buffer, generate the rest from scratch
        if buffer_replay_prob:
            n_new = np.random.binomial(sample_size, 1 - buffer_replay_prob)
            rand_inputs = self.generate_rand_inputs(n_new, self.inputs_shape, self.inputs_value_range)
            old_inputs = torch.cat(random.choices(self.buffer, k=sample_size - n_new), dim=0)
            starting_samples = torch.cat([rand_inputs, old_inputs], dim=0).detach().to(device)
        else:
            starting_samples = (
                self.generate_rand_inputs(sample_size, self.inputs_shape, self.inputs_value_range).detach().to(device)
            )
        # perform MCMC sampling
        samples = SGLDSampler.generate_samples(
            model=self.model,
            num_steps=num_steps if num_steps is not None else self.num_steps,
            step_size=step_size if step_size is not None else self.step_size,
            noise_eps=noise_eps if noise_eps is not None else self.noise_eps,
            grad_clamp=grad_clamp if grad_clamp is not None else self.grad_clamp,
            energy_function=energy_function if energy_function is not None else self.energy_function,
            inputs_value_range=inputs_value_range if inputs_value_range is not None else self.inputs_value_range,
            init_inputs=starting_samples,
            return_samples_per_step=return_samples_per_step,
        )

        # add new inputs to the buffer and remove old ones if needed
        if update_buffer:
            self.buffer = list(samples.to(torch.device("cpu")).chunk(sample_size, dim=0)) + self.buffer
            self.buffer = self.buffer[: self.buffer_size]
        return samples

    @staticmethod
    def generate_samples(
        model,
        init_inputs,
        energy_function: th.Optional[th.Callable] = None,
        num_steps: int = 60,
        step_size: int = 10,
        noise_eps: float = 0.005,
        inputs_value_range: th.Optional[th.Union[tuple, float]] = (-1.0, 1.0),
        grad_clamp: th.Optional[th.Union[tuple, float]] = (-0.03, 0.03),
        return_samples_per_step: th.Optional[th.Union[int, bool]] = False,
        force_eval: bool = True,
        **kwargs
    ):
        """
        Sample inputs from a given model.
        Args:
            model: Neural network to use for modeling E_theta
            init_inputs:inputs to start from for sampling.
            num_steps (int): Number of iterations in the MCMC algorithm.
            step_size (int): Learning rate nu of SGLD algorithm
            noise_eps (float, optional): Brownian motion noise std
            grad_clamp (tuple, int, optional): Clamp bounds for step gradients
            samples_clamp (tuple, int, optional): Clamp bounds generated samples
            return_samples_per_step (bool): If True, we return the sample at every iteration of the MCMC
        """
        # freeze model parameters and freeze model state (for compute efficiency)
        if force_eval:
            is_training = model.training
            model.eval()
        old_param_states = torchde.training.utils.freeze_params(model)

        samples = init_inputs
        samples.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Use a buffer tensor in which we generate noise each loop iteration.
        # Which is more efficient than creating a new tensor every iteration.
        if noise_eps:
            noise = torch.randn(samples.shape, device=samples.device)

        # List for storing generations at each step (for later analysis)
        samples_per_step = []

        # Loop over K (num_steps)
        for i in range(num_steps):
            # Part 1: Add noise to the input.
            if noise_eps:
                noise.normal_(0, noise_eps)
                samples.data.add_(noise.data)
                if inputs_value_range:
                    samples.data.clamp_(inputs_value_range[0], inputs_value_range[1])

            # Part 2: calculate gradients for the current input.
            if energy_function is None:
                out_imgs = -model(samples, **kwargs)
            else:
                out_imgs = energy_function(samples, model=model, **kwargs)
            out_imgs.sum().backward()
            if grad_clamp:
                # For stabilizing and preventing too high gradients
                samples.grad.data.clamp_(*(grad_clamp if isinstance(grad_clamp, tuple) else (-grad_clamp, grad_clamp)))

            # Apply gradients to our current samples
            samples.data.add_(-step_size * samples.grad.data)
            samples.grad.detach_()
            samples.grad.zero_()
            if inputs_value_range:
                samples.data.clamp_(inputs_value_range[0], inputs_value_range[1])

            # keeping all generated steps could be memory expensive, so we return the samples at every return_samples_per_step steps
            # or all of them in case return_samples_per_step is True
            if (isinstance(return_samples_per_step, bool) and return_samples_per_step) or (
                return_samples_per_step and not (i % return_samples_per_step)
            ):
                samples_per_step.append(samples.clone().detach())

        # Reactivate gradients for parameters for training
        torchde.training.utils.unfreeze_params(model, old_param_states)
        if force_eval:
            model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_samples_per_step:
            return torch.stack(samples_per_step, dim=0)
        else:
            return samples
