import torch
import typing as th
from torchde.utils import FunctionDescriptor
from torchde.training.module import DETrainingModule
from torchde.training.criterion import Criterion
from torchde.training.sgld.sampler import SGLDSampler
from torchde.training.module import DETrainingModule


class SGLDTrainingModule(DETrainingModule):
    def __init__(
        self,
        # model
        model: th.Optional[torch.nn.Module] = None,
        model_cls: th.Optional[str] = None,
        model_args: th.Optional[dict] = None,
        anomaly_detector_score: th.Optional[th.Union[str, FunctionDescriptor]] = None,
        # sampler
        sampler_args: th.Optional[dict] = None,
        # criterion
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
            anomaly_detection_score=anomaly_detector_score,
            criterion="torchde.training.sgld.criterion.SGLDTrainingCriterion",
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
        self.sampler = SGLDSampler(model=self.model, **(sampler_args or self.hparams.sampler_args or {}))

    def step(
        self,
        batch: th.Optional[th.Any] = None,
        batch_idx: th.Optional[int] = None,
        optimizer_idx: th.Optional[int] = None,
        name: str = "train",
        inputs: th.Optional[th.Any] = None,
        labels: th.Optional[th.Any] = None,
        transform_inputs: bool = True,
        transform_labels: bool = True,
        return_results: bool = False,
        return_factors: bool = False,
        log_results: bool = True,
        **kwargs,  # additional arguments to pass to the criterion and attacker
    ):
        inputs, labels = self.process_inputs(
            batch, inputs=inputs, labels=labels, transform_inputs=transform_inputs, transform_labels=transform_labels
        )
        samples = self.sampler.sample(sample_size=inputs.shape[0], update_buffer=name == "val", device=inputs.device)
        return super().step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer_idx=optimizer_idx,
            name=name,
            inputs=inputs,
            labels=labels,
            transform_inputs=False,
            transform_labels=False,
            return_results=return_results,
            return_factors=return_factors,
            log_results=log_results,
            samples=samples,
            **kwargs,
        )
