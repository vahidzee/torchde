import torch
import typing as th
import functools
from torchde.utils import process_function_description
from .layers import OrderedBlock, AutoRegressiveDensityEstimator1D


class MADE(torch.nn.Module):
    """
    Masked Autoregressive (Auto-encoder) Density Estimator (MADE)

    Attributes:
        in_features: The number of input features.
        num_masks: The number of masks to use. (0 for infinite masks)
        layers: Hidden layers of the MADE. (list of number of features)
        bias: If True, adds a bias term to linear computations.
        residual: If True, adds a residual connections.
        residual_scale: If True, scales the residual connection by the number of connections.
        residual_factor: factor to scale the residual connection by.
        residual_masked_connections: If True, adds a residual connection to every valid label.
        activation: The activation function to use (None for linear model).
        activation_args: The arguments to pass to the activation function.
        batch_norm: If True, adds a batch normalization layer in blocks.
        batch_norm_args: The arguments to pass to the batch normalization layers.
        num_mixtures: The number of mixtures to use in the density estimator.
        share_params_features: The number of features to share the parameters of the density estimator.
        distribution:
            The distribution to use in the density estimator. (subclass of torch.distributions.Distribution)
        distribution_args: The arguments to pass to the distribution class.
        distribution_params_transforms: The transforms to apply to the parameters of the distribution.
        bias_distribution: If True, adds bias terms to the density estimator.
        seed: The seed to use for the random number generator.
        masks_kind: The kind of masks and orderings to use. ("random" or "repeat") (default: random)
        device: The device to use.
        dtype: The data type to use.
        safe_grad_hook: The function to use to hook the gradients (right after forward).
    """

    def __init__(
        self,
        in_features: int,
        num_masks: int,
        # blocks
        layers: th.List[int],
        bias: bool = True,
        residual: bool = False,
        residual_scale: bool = True,
        residual_masked_connections: bool = True,
        residual_factor: float = 1.0,
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        # distribution
        num_mixtures: int = 1,
        share_params_features: th.Optional[int] = None,
        distribution: str = "torch.distributions.Normal",
        distribution_args: th.Optional[dict] = None,
        distribution_params_transforms: th.Optional[dict] = None,
        bias_distribution: bool = True,
        # general
        seed: int = 0,
        masks_kind: str = "random",
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # grad safety
        safe_grad_hook: str = "lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad)",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_masks = num_masks
        self.masks_kind = masks_kind
        self.__mask_indicator = 0
        self.seed = seed
        self.current_seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.residual = residual
        self.layers = torch.nn.Sequential(
            *[
                OrderedBlock(
                    in_features=in_features if not i else layers[i - 1],
                    out_features=layers[i],
                    bias=bias,
                    residual=residual,
                    residual_scale=residual_scale,
                    residual_masked_connections=residual_masked_connections,
                    residual_factor=residual_factor,
                    activation=activation,
                    activation_args=activation_args,
                    batch_norm=batch_norm,
                    batch_norm_args=batch_norm_args,
                    device=device,
                    dtype=dtype,
                )
                for i in range(len(layers))
            ]
        )
        self.density_estimator = AutoRegressiveDensityEstimator1D(
            dims_count=in_features,
            in_features=layers[-1],
            bias=bias_distribution,
            num_mixtures=num_mixtures,
            share_params_features=share_params_features,
            distribution=distribution,
            distribution_args=distribution_args,
            distribution_params_transforms=distribution_params_transforms,
            device=device,
            dtype=dtype,
        )

        self.register_buffer("ordering", torch.arange(self.in_features, dtype=torch.int, device=device))
        self.safe_grad_hook = safe_grad_hook

        self.reorder(initialization=True)

    @functools.cached_property
    def safe_grad_hook_function(self):
        return process_function_description(self.safe_grad_hook, entry_function="hook")

    def check_autoregressive_property(self, missing_deps: bool = True, extra_deps: bool = True, mask_index: int = 0):
        """
        Checks if the MADE is autoregressive.

        Args:
            missing_deps: If True, checks if the MADE is missing dependencies.
            extra_deps: If True, checks if the MADE has extra dependencies.
            mask_index: The index of the mask to use.

        Returns:
            None
        """
        inputs = torch.ones(1, self.in_features)
        inputs.requires_grad = True

        # force evaluation mode
        model_training = self.training
        self.eval()

        # reorder to specified mask_index
        current_mask = self.__mask_indicator
        self.reorder(mask_index=mask_index)
        outputs = self.log_prob(inputs, mask_index=mask_index)[0]
        outputs = outputs.squeeze(0)
        inputs.reshape(-1)

        for idx in range(outputs.shape[0]):
            # backpropogate to inputs to find data flow dependencies
            label = self.ordering[idx].item()
            outputs[idx].backward(retain_graph=True)  # reuse graph to avoid O(n^2) forward passes
            deps_idxs = torch.where(inputs.grad[0])[0]  # get data flow dependencies (non-zero gradients)
            deps_labels = set(self.ordering[deps_idxs].tolist())  # get labels of dependencies

            if missing_deps:  # check for missing dependencies (i.e. labels smaller or equal to currentlabel)
                missing = [i for i in range(label + 1) if i not in deps_labels]
                if missing:
                    print(f"output({idx})[label={label}] -> not depended on labels: {missing}")
            if extra_deps:  # check for extra dependencies (larger than current label)
                extra = [i for i in range(label + 1, self.in_features) if i in deps_labels]
                if extra:
                    print(f"output({idx})[label={label}] -> extra dependancies on labels: {extra}")
            inputs.grad.zero_()  # restart gradient calculation

        # resotre training mode
        if model_training:
            self.train()
        # restore ordering
        self.reorder(current_mask)

    def reorder(self, mask_index=None, initialization=False):
        """
        Either reorders to the next mask or to the specified mask.

        Args:
            mask_index: The index of the mask to use.
            initialization: If True, the mask is being initialized for the first time.

        Returns:
            None
        """
        # if there is only one mask, do nothing except if initialization is True (to initialize layers' orderings)
        if self.num_masks == 1 and not initialization:
            return

        # if mask_index is None, use next mask else use specified mask (and remember the original ordering)
        if self.__mask_indicator or mask_index is not None:
            # each ordering corresponds to a random number generotor and a random seed
            self.generator = torch.Generator().manual_seed(
                self.seed + (mask_index if mask_index is not None else self.__mask_indicator)
            )
            if mask_index is not None:  # if mask_index is not None, we are initializing the mask[mask_index]
                self.__mask_indicator = mask_index

        # initialize inputs ordering (mask_index=0 is cannonical ordering the rest are random permutations)
        if not self.num_masks or self.__mask_indicator:
            self.ordering.data.copy_(torch.randperm(self.in_features, generator=self.generator))
        elif not self.__mask_indicator:
            self.ordering.data.copy_(torch.arange(self.in_features, dtype=torch.int, device=self.ordering.device))
            self.generator = torch.Generator().manual_seed(self.seed)

        for i, layer in enumerate(self.layers):
            layer.reorder(
                inputs_ordering=self.ordering if not i else self.layers[i - 1].ordering,
                generator=self.generator,
                # force ordering to be the same for all layers if masks_kind is repeat
                ordering=self.ordering if self.masks_kind == "repeat" else None,
                allow_detached_neurons=self.residual,
                highest_ordering_label=self.in_features,
            )
        self.density_estimator.reorder(inputs_ordering=self.layers[-1].ordering, ordering=self.ordering)

        # if mask_index is None, move to next mask for the next model.reorder() call
        if mask_index is None and not initialization:
            self.__mask_indicator = ((self.__mask_indicator + 1) % self.num_masks) if self.num_masks else 0

    def forward(self, inputs, mask_index: th.Optional[int] = None, safe_grad: bool = True):
        if mask_index is None:
            results = self.density_estimator(self.layers(inputs))
            if results.requires_grad and safe_grad:
                results.register_hook(self.safe_grad_hook_function)
            return results

    def distributions(self, inputs):
        return self.density_estimator.distributions(self(inputs))

    def log_prob(self, inputs, reduce=False, mask_index=None):
        if mask_index is not None:
            current_ordering = self.__mask_indicator
            self.reorder(mask_index=mask_index)
            results = self.density_estimator.log_prob(inputs=inputs, params_logits=self(inputs), reduce=reduce)
            self.reorder(current_ordering)
            return results

        results = 0.0
        for i in range(self.num_masks or 1):
            self.reorder()
            results += self.density_estimator.log_prob(inputs=inputs, params_logits=self(inputs), reduce=reduce)
        results /= self.num_masks or 1.0
        return results

    def sample(self, num_samples=1, mask_index=None):
        results = torch.rand(num_samples, self.in_features, device=self.ordering.device)
        is_training = self.training
        self.eval()
        if mask_index is not None:
            current_mask = self.__mask_indicator
            self.reorder(mask_index=mask_index)
        with torch.no_grad():
            for i in range(self.in_features):
                results = self.distributions(results).sample()
        if mask_index is not None:
            self.reorder(mask_index=current_mask)
        if is_training:
            self.train()
        return results

    def extra_repr(self):
        return "num_masks={}, seed={}".format(
            self.num_masks or "inf",
            self.seed,
        )
