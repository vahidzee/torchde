import torch
import typing as th
import functools
import inspect
from .ordered_linear import OrderedLinear
from torchde.utils import process_function_description, get_value


class AutoRegressiveDensityEstimator1D(OrderedLinear):
    def __init__(
        self,
        dims_count: int,
        in_features: int,
        bias: bool = True,
        num_mixtures: int = 1,
        share_params_features: th.Optional[int] = None,
        distribution: str = "torch.distributions.Normal",
        distribution_args: th.Optional[dict] = None,
        distribution_params_transforms: th.Optional[dict] = None,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        self.num_mixtures = num_mixtures
        self.__distribution_name = distribution
        self.distribution = get_value(distribution) if isinstance(distribution, str) else distribution
        assert issubclass(
            self.distribution, torch.distributions.Distribution
        ), "expected a subclass of `torch.distributions.Distribution` as the estimator's distribution"
        self.distribution = functools.partial(self.distribution, **(distribution_args or dict()))
        self.share_params_features = share_params_features
        self.dims_count = dims_count
        super().__init__(
            in_features=in_features,
            out_features=(
                dims_count
                * (len(self.distribution_params_names) + (1 if self.num_mixtures > 1 else 0))
                * self.num_mixtures
            )
            if share_params_features is None
            else (share_params_features * dims_count),
            bias=bias,
            device=device,
            dtype=dtype,
            auto_connection=False,
        )

        # shared linear layer to compute dims' distribution parameters from features
        self.params_compute = (
            torch.nn.Linear(
                in_features=share_params_features,
                out_features=(len(self.distribution_params_names) + (1 if self.num_mixtures > 1 else 0))
                * self.num_mixtures,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            if share_params_features is not None
            else None
        )
        # outsourced functions to transform distribution parameters
        self.distribution_params_transforms = distribution_params_transforms

    @functools.cached_property
    def distribution_params_transforms_functions(self):
        return {
            name: process_function_description(value, "transform")
            for name, value in (self.distribution_params_transforms or dict()).items()
        }

    @functools.cached_property
    def distribution_params_names(self):
        params = inspect.signature(self.distribution).parameters
        return [
            name
            for name in params
            if name not in ["probs", "validate_args"]
            and (params[name].default is None or params[name].default is inspect._empty)
        ]

    @functools.cached_property
    def parameter_indeces(self):
        result = torch.concat(
            [
                (
                    (
                        torch.arange(self.dims_count, dtype=torch.long)
                        * (len(self.distribution_params_names) + (1 if self.num_mixtures > 1 else 0))
                        * self.num_mixtures
                    )[:, None]
                    + torch.arange(self.num_mixtures, dtype=torch.long)
                    + (index * self.num_mixtures)
                ).unsqueeze(0)
                for index in range(len(self.distribution_params_names) + (1 if self.num_mixtures > 1 else 0))
            ],
            dim=0,
        )
        return result if self.num_mixtures > 1 else result.squeeze(-1)

    def transform_distribution_parameters(self, params_logits) -> th.Dict[str, torch.Tensor]:
        "transforms distribution parameters given their raw logits (output of model)"
        results = {
            name: (
                params_logits[:, i]
                if name not in self.distribution_params_transforms_functions
                else self.distribution_params_transforms_functions[name](params_logits[:, i])
            )
            for i, name in enumerate(self.distribution_params_names)
        }
        if self.num_mixtures > 1:
            results["mixture_logits"] = (
                self.distribution_params_transforms_functions["mixture_logits"](params_logits[:, -1])
                if "mixture_logits" in self.distribution_params_transforms_functions
                else params_logits[:, -1]
            )
        return results

    def distributions(self, params_logits, params=None) -> torch.distributions.Distribution:
        params = params or self.transform_distribution_parameters(params_logits)
        component_params = {key: value for key, value in params.items() if key != "mixture_logits"}
        component_distributions = self.distribution(**(component_params))
        if self.num_mixtures == 1:
            return component_distributions
        mixture_distributions = torch.distributions.Categorical(logits=(params["mixture_logits"]))
        return torch.distributions.MixtureSameFamily(mixture_distributions, component_distributions)

    def log_prob(self, inputs, params_logits, params=None, reduce=False) -> torch.Tensor:
        log_probs = self.distributions(params_logits=params_logits, params=params).log_prob(inputs)
        return log_probs.sum(-1) if reduce else log_probs

    def forward(self, input_features) -> torch.Tensor:
        "Computes the untransformed parameter logits, given ordered input_features"
        if self.params_compute is None:
            return super().forward(input_features)[:, self.parameter_indeces]

        features = super().forward(input_features).reshape(input_features.shape[0], self.dims_count, -1)
        return self.params_compute(features).reshape(input_features.shape[0], -1)[:, self.parameter_indeces]

    def extra_repr(self):
        return "distribution={}, num_mixtures={}{}".format(
            self.__distribution_name,
            self.num_mixtures,
            f", params_features={self.share_params_features}" if self.share_params_features else "",
        )
