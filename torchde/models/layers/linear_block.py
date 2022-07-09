import typing as th
import torch
from torchde.utils import get_value


class LinearBlock(torch.nn.Module):
    def __init__(
        self,
        # linear args
        in_features: int,
        out_features: int,
        bias: bool = True,
        # activation
        activation: th.Optional[str] = "torch.nn.GELU",
        activation_args: th.Optional[dict] = None,
        # batch norm
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        # residual
        residual: bool = True,
        residual_factor: float = 1.0,
        # general parameters
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.activation = get_value(activation)(**(activation_args or dict())) if activation else None
        self.batch_norm = (
            torch.nn.BatchNorm1d(num_features=out_features, dtype=dtype, device=device, **(batch_norm_args or dict()))
            if batch_norm
            else None
        )
        self.residual, self.residual_factor = residual, residual_factor

    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.batch_norm(outputs) if self.batch_norm else outputs
        if self.residual and self.residual_factor and outputs.shape[-1] == inputs.shape[-1]:
            outputs = outputs + self.residual_factor * inputs
        return outputs
