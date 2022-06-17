import torch
import typing as th
from .layers import OrderedBlock, AutoRegressiveDensityEstimator1D


class MADE(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_masks: int,
        # blocks
        layers: th.List[int],
        bias: bool = True,
        residual: bool = False,
        residual_scale: bool = True,
        activation: str = "torch.nn.GELU",
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
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_masks = num_masks
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

        self.register_buffer(
            "ordering", torch.arange(self.in_features, dtype=torch.int, device=device)
        )

        self.reorder(initialization=True)

    def reorder(self, mask_index=None, initialization=False):
        if self.num_masks == 1 and not initialization:
            return
        if self.__mask_indicator or mask_index is not None:
            self.generator = torch.Generator().manual_seed(
                self.seed
                + (mask_index if mask_index is not None else self.__mask_indicator)
            )
            if mask_index is not None:
                self.__mask_indicator = mask_index
        if not self.num_masks or self.__mask_indicator:

            self.ordering.data.copy_(
                torch.randperm(self.in_features, generator=self.generator)
            )

        if not self.__mask_indicator:
            self.ordering.data.copy_(
                torch.arange(
                    self.in_features, dtype=torch.int, device=self.ordering.device
                )
            )
            self.generator = torch.Generator().manual_seed(self.seed)

        for i, layer in enumerate(self.layers):
            layer.reorder(
                inputs_ordering=self.ordering if not i else self.layers[i - 1].ordering,
                generator=self.generator,
                allow_detached_neurons=self.residual,
                highest_ordering_label=self.in_features,
            )
        self.density_estimator.reorder(
            inputs_ordering=self.layers[-1].ordering, ordering=self.ordering
        )
        if mask_index is None and not initialization:
            self.__mask_indicator = (
                ((self.__mask_indicator + 1) % self.num_masks) if self.num_masks else 0
            )

    def forward(
        self, inputs, mask_index: th.Optional[int] = None, safe_grad: bool = True
    ):
        if mask_index is None:
            results = self.density_estimator(self.layers(inputs))
            if results.requires_grad and safe_grad:
                results.register_hook(
                    lambda grad: torch.where(
                        torch.isnan(grad) + torch.isinf(grad),
                        torch.zeros_like(grad),
                        grad,
                    )
                )
            return results

    def distributions(self, inputs):
        return self.density_estimator.distributions(self(inputs))

    def log_prob(self, inputs, reduce=False, mask_index=None):
        if mask_index is not None:
            current_ordering = self.__mask_indicator
            self.reorder(mask_index=mask_index)
            results = self.density_estimator.log_prob(
                inputs=inputs, params_logits=self(inputs), reduce=reduce
            )
            self.reorder(current_ordering)
            return results

        results = 0.0
        for i in range(self.num_masks or 1):
            self.reorder()
            results += self.density_estimator.log_prob(
                inputs=inputs, params_logits=self(inputs), reduce=reduce
            )
        results /= self.num_masks or 1.0
        return results

    def sample(self, num_samples=1, generator=None, mask_index=None):
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
