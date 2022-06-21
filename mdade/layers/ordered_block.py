import typing as th
import torch
from .ordered_linear import OrderedLinear
from .ordered_residual import OrderedResidual1D
from ..utils import get_value


class OrderedBlock(torch.nn.Module):
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
        residual_scale: bool = True,
        # ordering args
        auto_connection: bool = True,
        # general parameters
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.linear = OrderedLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            auto_connection=auto_connection,
            device=device,
            dtype=dtype,
        )
        self.activation = get_value(activation)(**(activation_args or dict())) if activation else None
        self.batch_norm = (
            torch.nn.BatchNorm1d(num_features=out_features, dtype=dtype, device=device, **(batch_norm_args or dict()))
            if batch_norm
            else None
        )
        self.residual = (
            OrderedResidual1D(
                in_features=in_features,
                out_features=out_features,
                auto_connection=auto_connection,
                device=device,
                scale=residual_scale,
            )
            if residual
            else None
        )

    @property
    def ordering(self):
        return self.linear.ordering

    def reorder(
        self,
        inputs_ordering: torch.IntTensor,
        ordering: th.Optional[torch.IntTensor] = None,
        allow_detached_neurons: bool = True,
        highest_ordering_label: th.Optional[int] = None,
        generator: th.Optional[torch.Generator] = None,
    ):
        self.linear.reorder(
            inputs_ordering=inputs_ordering,
            ordering=ordering,
            allow_detached_neurons=allow_detached_neurons,
            highest_ordering_label=highest_ordering_label,
            generator=generator,
        )

        if self.residual:
            self.residual.reorder(
                ordering=self.linear.ordering,
                inputs_ordering=inputs_ordering,
            )

    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.batch_norm(outputs) if self.batch_norm else outputs
        outputs = self.residual(inputs, outputs) if self.residual else outputs
        return outputs
