import torch
import typing as th
from .ordering_mixin import OrderedLayerMixin1D


class OrderedResidual1D(torch.nn.Module, OrderedLayerMixin1D):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: th.Optional[torch.device] = None,
        auto_connection: bool = True,
        scale=True,
        masked_connections=True,
        factor: float = 1.0,
    ) -> None:
        super().__init__()
        OrderedLayerMixin1D.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            device=device,
            mask_dtype=torch.float,
            auto_connection=auto_connection,
        )
        self.scale = scale
        self.masked_connection = masked_connections
        self.factor = factor

    def reorder(self, *args, **kwargs):
        OrderedLayerMixin1D.reorder(self, *args, **kwargs)
        if self.scale:
            mask = torch.div(self.mask, self.mask.sum(1, keepdim=True))
            self.mask.data.copy_(torch.where(mask != mask, torch.zeros_like(mask), mask))

    def forward(self, in_features, target_features) -> torch.Tensor:
        if self.masked_connection:
            return target_features + in_features @ self.mask.T * self.factor
        return target_features + (in_features * self.factor if in_features.shape == target_features.shape else 0)
