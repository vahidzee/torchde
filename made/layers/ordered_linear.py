import torch
import typing as th
from .ordering_mixin import OrderedLayerMixin1D


class OrderedLinear(
    torch.nn.Linear,
    OrderedLayerMixin1D,
):
    """
    Linear layer with ordered outputs to maintain an autoregressive data flow.

    Attributes:
        ordering:
            Current ordering of the output neurons.
        mask:
            A zero-one 2D mask matrix defining the connectivity of output and
            input neurons.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        auto_connection: bool = True,
    ) -> None:
        """
        Initializes module by initializing an ordering mixin and a linear layer.

        Args:
            in_features: Number of input feature dimensions in 1D.
            out_features: Number of output feature dimensions in 1D.
            bias: Whether to use a bias vector for the linear operation.
            device: The destination device for the buffers and parameters.
            dtype: Data type for linear layer parameters.
            masked_dtype: Data type for mask matrix.
            auto_connection: Whether to allow equal label connections.

        Returns:
            None
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        OrderedLayerMixin1D.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            device=device,
            auto_connection=auto_connection,
        )

    def forward(self, inputs: torch.Tensor):
        """
        Computes masked linear operation.

        Args:
            inputs: Input tensor (batched in the first dimensions.)

        Returns:
            A `torch.Tensor` which equals to masked linear operation on inputs, or:
                `inputs @ (self.mask * self.weights).T + self.bias`
        """
        return torch.nn.functional.linear(inputs, self.mask * self.weight, self.bias)
