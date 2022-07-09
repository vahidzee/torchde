import torch
import typing as th
import functools
from torchde.utils import process_function_description
from .layers import LinearBlock


class MLP(torch.nn.Module):
    """
    Multi-layer Perceptron (MLP) Network.

    Attributes:
        in_features: The number of input features.
        layers: Hidden layers of the MLP. (list of number of features)
        bias: If True, adds a bias term to linear computations.
        residual: If True, adds a residual connections.
        residual_factor: factor to scale the residual connection by (defaults to 1.0).
        activation: The activation function to use (None for linear model).
        activation_args: The arguments to pass to the activation function.
        batch_norm: If True, adds a batch normalization layer in blocks.
        batch_norm_args: The arguments to pass to the batch normalization layers.
        device: The device to use.
        dtype: The data type to use.
        safe_grad_hook: The function to use to hook the gradients (right after forward).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # blocks
        layers: th.List[int],
        bias: bool = True,
        residual: bool = False,
        residual_factor: float = 1.0,
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        # general
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # grad safety
        safe_grad_hook: str = "lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad)",
    ):
        super().__init__()
        self.in_features = in_features

        self.layers = torch.nn.Sequential(
            *[
                LinearBlock(
                    in_features=in_features if not i else layers[i - 1],
                    out_features=layers[i],
                    bias=bias,
                    residual=residual,
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
        self.final_layer = torch.nn.Linear(
            in_features=layers[-1], out_features=out_features, bias=bias, device=device, dtype=dtype
        )
        self.safe_grad_hook = safe_grad_hook

    @functools.cached_property
    def safe_grad_hook_function(self):
        return process_function_description(self.safe_grad_hook, entry_function="hook")

    def forward(self, inputs, safe_grad: bool = True):
        results = self.final_layer(self.layers(inputs))
        if results.requires_grad and safe_grad:
            results.register_hook(self.safe_grad_hook_function)
        return results
