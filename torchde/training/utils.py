import torch
import typing as th


def freeze_params(model: torch.nn.Module):
    old_states = []
    for param in model.parameters():
        old_states.append(param.requires_grad)
        param.requires_grad = False
    return old_states


def unfreeze_params(model: torch.nn.Module, old_states: th.Optional[th.List[bool]] = None):
    for idx, param in enumerate(model.parameters()):
        param.requires_grad = True if old_states is None else old_states[idx]
