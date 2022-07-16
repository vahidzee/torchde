import torch
import typing as th
import torchde.utils
import torchde.models.utils


class VisionBackbone(torch.nn.Module):
    def __init__(
        self,
        backbone_cls: str,
        backbone_args: th.Optional[dict] = None,
        in_channels: th.Optional[int] = None,
        out_features: th.Optional[int] = None,
        first_conv_layer_name: th.Optional[str] = None,
        first_conv_kernel_size: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
        first_conv_padding: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
        first_conv_stride: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
        first_conv_bias: th.Optional[bool] = None,
        probe_layer_name: th.Optional[str] = None,
        probe_bias: th.Optional[bool] = None,
        probe_transfer_weights: bool = False,
    ):
        super().__init__()
        self.model = torchde.utils.get_value(backbone_cls)(**(backbone_args or {}))
        if first_conv_layer_name is not None:
            torchde.utils.set_value(
                first_conv_layer_name,
                value=torchde.models.utils.transfer_conv2d_layer(
                    torchde.utils.get_value(first_conv_layer_name, context=self.model),
                    in_channels=in_channels,
                    kernel_size=first_conv_kernel_size,
                    padding=first_conv_padding,
                    stride=first_conv_stride,
                    bias=first_conv_bias,
                ),
                context=self.model,
            )
        if probe_layer_name is not None:
            torchde.utils.set_value(
                probe_layer_name,
                value=torchde.models.utils.transfer_linear_layer(
                    torchde.utils.get_value(probe_layer_name, context=self.model),
                    out_features=out_features,
                    bias=probe_bias,
                    transfer_weights=probe_transfer_weights,
                ),
                context=self.model,
            )

    def forward(self, x):
        return self.model(x)
