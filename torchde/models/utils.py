import torch
import typing as th


def transfer_conv2d_layer(
    old_conv2d: torch.nn.Conv2d,
    in_channels: th.Optional[int] = None,
    out_channels: th.Optional[int] = None,
    bias: th.Optional[bool] = None,
    padding: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
    kernel_size: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
    stride: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
):
    """
    Create a new conv2d layer with given specifications and transfer weights from the old conv2d layer
    """
    in_channels = old_conv2d.in_channels if in_channels is None else in_channels
    out_channels = old_conv2d.out_channels if out_channels is None else out_channels
    bias = (old_conv2d.bias is not None) if bias is None else bias
    padding = old_conv2d.padding if padding is None else padding
    kernel_size = old_conv2d.kernel_size if kernel_size is None else kernel_size
    stride = old_conv2d.stride if stride is None else stride
    if (
        in_channels == old_conv2d.in_channels
        and out_channels == old_conv2d.out_channels
        and bias == (old_conv2d.bias is not None)
        and kernel_size == old_conv2d.kernel_size
    ):
        old_conv2d.padding = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        old_conv2d.stride = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        return old_conv2d

    new_conv2d = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        bias=bias,
        padding=padding,
        kernel_size=kernel_size,
        stride=stride,
    )
    new_conv2d.weight[
        : min(new_conv2d.out_channels, old_conv2d.out_channels),
        : min(new_conv2d.in_channels, old_conv2d.in_channels),
        : min(new_conv2d.kernel_size[0], old_conv2d.kernel_size[0]),
        : min(new_conv2d.kernel_size[1], old_conv2d.kernel_size[1]),
    ].data = old_conv2d.weight[
        : min(new_conv2d.out_channels, old_conv2d.out_channels),
        : min(new_conv2d.in_channels, old_conv2d.in_channels),
        : min(new_conv2d.kernel_size[0], old_conv2d.kernel_size[0]),
        : min(new_conv2d.kernel_size[1], old_conv2d.kernel_size[1]),
    ].data
    if old_conv2d.bias is not None and new_conv2d.bias is not None:
        new_conv2d.bias[: min(new_conv2d.out_channels, old_conv2d.out_channels)].data = old_conv2d.bias[
            : min(new_conv2d.out_channels, old_conv2d.out_channels)
        ].data
    return new_conv2d


def transfer_linear_layer(
    old_linear: torch.nn.Linear,
    in_features: th.Optional[int] = None,
    out_features: th.Optional[int] = None,
    bias: th.Optional[bool] = None,
    transfer_weights: th.Optional[bool] = True,
):
    """
    Create a new linear layer with given specifications and transfer weights from the old linear layer
    """
    in_features = old_linear.in_features if in_features is None else in_features
    out_features = old_linear.out_features if out_features is None else out_features
    bias = (old_linear.bias is not None) if bias is None else bias
    if (
        in_features == old_linear.in_features
        and out_features == old_linear.out_features
        and bias == (old_linear.bias is not None)
    ):
        return old_linear

    new_linear = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )
    if not transfer_weights:
        return new_linear
    new_linear.weight[
        : min(new_linear.out_features, old_linear.out_features),
        : min(new_linear.in_features, old_linear.in_features),
    ].data = old_linear.weight[
        : min(new_linear.out_features, old_linear.out_features),
        : min(new_linear.in_features, old_linear.in_features),
    ].data
    if old_linear.bias is not None and new_linear.bias is not None:
        new_linear.bias[: min(new_linear.out_features, old_linear.out_features)].data = old_linear.bias[
            : min(new_linear.out_features, old_linear.out_features)
        ].data
    return new_linear
