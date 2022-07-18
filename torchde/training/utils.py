import torch
import typing as th
import itertools


def freeze_params(
    model: th.Optional[torch.nn.Module] = None,
    optimizer: th.Optional[torch.optim.Optimizer] = None,
):
    params = (
        model.parameters()
        if model is not None
        else itertools.chain(param  for i in range(len(optimizer.param_groups)) for param in optimizer.param_groups[i]["params"])
    )
    old_states = []
    for param in params:
        old_states.append(param.requires_grad)
        param.requires_grad = False
    return old_states

def unfreeze_params(
    model: th.Optional[torch.nn.Module] = None,
    optimizer: th.Optional[torch.optim.Optimizer] = None,
    old_states: th.Optional[th.List[bool]] = None,
):
    params = (
        model.parameters()
        if model is not None
        else itertools.chain(param  for i in range(len(optimizer.param_groups)) for param in optimizer.param_groups[i]["params"])
    )
    for idx, param in enumerate(params):
        param.requires_grad = True if old_states is None else old_states[idx]