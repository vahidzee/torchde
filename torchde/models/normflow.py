import torch
import typing as th
import torchde.utils


def create_nf(
    in_features: int,
    flow_type: str = "realnvp",
    num_layers: int = 8,
    latent_size: th.Optional[int] = None,
    q0: str = "normflow.distributions.DiagGaussian",
    q0_args: th.Optional[dict] = None,
):
    import normflow as nf

    latent_size = latent_size or in_features * 4
    flows = []
    b = torch.tensor([0 if i < in_features // 2 else 1 for i in range(in_features)])
    for i in range(num_layers):
        if flow_type == "planar":
            flows += [nf.flows.Planar((in_features,))]
        elif flow_type == "radial":
            flows += [nf.flows.Radial((in_features,))]
        elif flow_type == "nice":
            flows += [
                nf.flows.MaskedAffineFlow(
                    b, nf.nets.MLP([in_features, latent_size, latent_size, in_features], init_zeros=True)
                )
            ]
        elif flow_type == "realnvp":
            flows += [
                nf.flows.MaskedAffineFlow(
                    b,
                    nf.nets.MLP([in_features, latent_size, latent_size, in_features], init_zeros=True),
                    nf.nets.MLP([in_features, latent_size, latent_size, in_features], init_zeros=True),
                )
            ]
        b = 1 - b  # parity alternation for mask
    q0 = torchde.utils.get_value(q0)(**(q0_args or {}))
    return nf.NormalizingFlow(flows=flows, q0=q0)
