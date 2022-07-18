import torch
import typing as th
import torchde.utils


class DummyCNN(torch.nn.Module):
    def __init__(
        self,
        inputs_shape: th.Union[list, tuple],
        latent_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bn: bool = True,
        bn_affine: bool = True,
        bn_eps: bool = True,
        bn_latent: bool = False,
        layers_activation: th.Optional[str] = "torch.nn.ReLU",
        latent_activation: th.Optional[str] = None,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = inputs_shape
        self.num_layers = num_layers
        self.num_input_channels = inputs_shape[0]
        self.latent_size = latent_size
        self.bn_latent = bn_latent
        self.latent_activation = latent_activation
        self.layers_activation = layers_activation
        self.input_size = min(inputs_shape[1], inputs_shape[2])
        self.bias = bias
        self.bn_affine = bn_affine
        self.bn_eps = bn_eps

        # convolutional blocks
        self.features = torch.nn.Sequential()

        assert num_layers > 0, "non-positive number of layers"
        latent_image_size = self.input_size // (2**num_layers)

        assert latent_image_size > 0, "number of layers is too large"

        # convolutional layers
        self.conv_layers = []
        for i in range(self.num_layers):
            layer = torch.nn.Sequential()
            conv = torch.nn.Conv2d(
                32 * (2 ** (i - 1)) if i else self.num_input_channels,
                32 * (2**i),
                5,
                bias=bias,
                padding=2,
                dtype=dtype,
                device=device,
            )
            layer.add_module("conv", conv)
            if bn:
                layer.add_module(
                    "batch_norm",
                    torch.nn.BatchNorm2d(32 * (2**i), eps=bn_eps, affine=bn_affine, dtype=dtype, device=device),
                )
            if layers_activation:
                layer.add_module("activation", torchde.utils.get_value(layers_activation)())
            layer.add_module("pool", torch.nn.MaxPool2d(2, 2))
            self.conv_layers.append(layer)

            self.features.add_module(f"layer{i}", layer)

        # fully connected layer
        self.fc_block = torch.nn.Sequential()
        self.fc_block.add_module(
            "fc",
            torch.nn.Linear(
                32 * (2 ** (num_layers - 1)) * (latent_image_size**2),
                self.latent_size,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
        )

        # fully connected activation
        if self.latent_activation is not None:
            self.fc_block.add_module(
                "activation", torchde.utils.get_value(latent_activation)() if latent_activation else None
            )

        # fully connected batch_norm
        if self.bn_latent:
            self.fc_block.add_module(
                "batch_norm",
                torch.nn.BatchNorm1d(self.latent_size, eps=bn_eps, affine=bn_affine, device=device, dtype=dtype),
            )

    def forward(self, inputs: torch.Tensor):
        return self.fc_block(self.features(inputs).reshape(inputs.shape[0], -1))
