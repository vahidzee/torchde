import typing as th
import torch
from torchde.utils import FunctionDescriptor
from torchde.training.module import DETrainingModule
from torchde.training.criterion import Criterion


class MADETrainingModule(DETrainingModule):
    def __init__(
        self,
        # model
        model: th.Optional[torch.nn.Module] = None,
        model_cls: th.Optional[str] = "torchde.models.MADE",
        model_args: th.Optional[dict] = None,
        anomaly_detector_score: th.Optional[th.Union[str, FunctionDescriptor]] = None,
        # criterion
        criterion: th.Union[Criterion, str] = "torchde.training.made.MADETrainingCriterion",
        criterion_args: th.Optional[dict] = None,
        # attacks
        attack_args: th.Optional[dict] = None,
        inputs_transform: th.Optional[FunctionDescriptor] = None,
        # optimization configs
        optimizer: str = "torch.optim.Adam",
        optimizer_args: th.Optional[dict] = None,
        lr: float = 1e-4,
        scheduler: th.Optional[str] = None,
        scheduler_args: th.Optional[dict] = None,
        scheduler_interval: str = "epoch",
        scheduler_frequency: int = 1,
        scheduler_monitor: th.Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            model_cls=model_cls,
            model_args=model_args,
            anomaly_detector_score=anomaly_detector_score,
            criterion=criterion,
            criterion_args=criterion_args,
            attack_args=attack_args,
            inputs_transform=inputs_transform,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            lr=lr,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_monitor=scheduler_monitor,
        )
