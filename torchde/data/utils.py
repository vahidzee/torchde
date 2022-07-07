import typing as th
from numpy import isin
import numpy as np
from torchvision import transforms as tv_transforms
from torchde.utils import get_value, process_function_description
import torch
import inspect


def initialize_transforms(transforms: th.Optional[th.Union[list, dict, th.Any]]):
    if transforms is None or (not inspect.isclass(transforms) and callable(transforms)):
        return transforms

    # list of other transforms
    if isinstance(transforms, list):
        return tv_transforms.Compose([initialize_transforms(i) for i in transforms])

    # either a class and args, or a code block and entry function
    if isinstance(transforms, dict):
        if "class_path" in transforms:
            return get_value(transforms["class_path"])(transforms.get("init_args", dict()))
        value = process_function_description(transforms, "transform")
        return value() if inspect.isclass(value) else value
    if isinstance(transforms, str):
        try:
            return get_value(transforms)()
        except:
            return process_function_description(transforms, "transform")


class AnomallyBaseDataset(torch.utils.data.Dataset):
    def __init__(self, normal_targets: th.Optional[None], relabel: bool = False):
        self.normal_targets = np.array(normal_targets if normal_targets is not None else [])
        self.relabel = relabel

    def __len__(self):
        return len(self.dataset) if not hasattr(self, "indices") else len(self.indices)

    def __getitem__(self, index):
        if hasattr(self, "indices"):
            index = self.indices[index]
        if self.relabel:
            inputs, target = self.dataset[index]
            return inputs, 1 if target in self.normal_targets else 0
        return self.dataset[index]


class NormalDataset(AnomallyBaseDataset):
    """Filters out the normal samples from the dataset."""

    def __init__(self, original_dataset: torch.utils.data.Dataset, normal_targets: th.Optional[th.List] = None):
        super().__init__(normal_targets, relabel=False)
        self.dataset = (
            original_dataset
            if not isinstance(original_dataset, torch.utils.data.torch.utils.data.Subset)
            else original_dataset.dataset
        )
        self.indices = np.array(
            np.arange(len(original_dataset))
            if not isinstance(original_dataset, torch.utils.data.Subset)
            else original_dataset.indices
        )
        if len(normal_targets):
            indices_map = np.zeros(len(self.dataset), dtype=np.int)
            indices_map[self.indices] = 1
            self.indices = np.where(indices_map & np.isin(self.dataset.targets, self.normal_targets))[0]


class IsNormalDataset(AnomallyBaseDataset):
    """Labels the samples as normal (1) or not normal. (0)"""

    def __init__(self, original_dataset: torch.utils.data.Dataset, normal_targets: th.Optional[th.List] = None):
        super().__init__(normal_targets, relabel=True)
        self.dataset = (
            original_dataset if not isinstance(original_dataset, torch.utils.data.Subset) else original_dataset.dataset
        )
        self.indices = np.array(
            np.arange(len(self.dataset))
            if not isinstance(original_dataset, torch.utils.data.Subset)
            else original_dataset.indices
        )
