import torch
import typing as th
from torchde.utils import FunctionDescriptor
from torchde.training.module import DETrainingModule
from torchde.training.sgld.sampler import SGLDSampler
from torchde.training.module import DETrainingModule


class SGLDTrainingModule(DETrainingModule):
    def __init__(
        self,
        # model
        model: th.Optional[torch.nn.Module] = None,
        model_cls: th.Optional[str] = None,
        model_args: th.Optional[dict] = None,
        anomaly_detection_score: th.Optional[th.Union[str, FunctionDescriptor]] = None,
        # sampler
        sampler_args: th.Optional[dict] = None,
        # criterion
        criterion_args: th.Optional[dict] = None,
        # attacks
        attack_args: th.Optional[dict] = None,
        # input transforms
        inputs_transform: th.Optional[FunctionDescriptor] = None,
        inputs_noise_eps: th.Optional[float] = None,
        labels_transform: th.Optional[FunctionDescriptor] = None,
        # optimization configs
        optimizer: str = "torch.optim.Adam",
        optimizer_is_active: th.Optional[th.Union[FunctionDescriptor, th.List[FunctionDescriptor]]] = None,
        optimizer_parameters: th.Optional[th.Union[th.List[str], str]] = None,
        optimizer_args: th.Optional[dict] = None,
        # learning rate
        lr: th.Union[th.List[float], float] = 1e-4,
        # schedulers
        scheduler: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_name: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_optimizer: th.Optional[th.Union[int, th.List[int]]] = None,
        scheduler_args: th.Optional[th.Union[dict, th.List[dict]]] = None,
        scheduler_interval: th.Union[str, th.List[str]] = "epoch",
        scheduler_frequency: th.Union[int, th.List[int]] = 1,
        scheduler_monitor: th.Optional[th.Union[str, th.List[str]]] = None,
        # instantiation configs
        save_hparams: bool = True,
    ) -> None:
        super().__init__(
            # model
            model=model,
            model_cls=model_cls,
            model_args=model_args,
            # anomaly detection
            anomaly_detection_score=anomaly_detection_score,
            # criterion
            criterion="torchde.training.sgld.criterion.SGLDTrainingCriterion",
            criterion_args=criterion_args,
            # attacks
            attack_args=attack_args,
            # input transforms
            inputs_transform=inputs_transform,
            inputs_noise_eps=inputs_noise_eps,
            labels_transform=labels_transform,
            # optimization configs
            optimizer=optimizer,
            optimizer_is_active=optimizer_is_active,
            optimizer_parameters=optimizer_parameters,
            optimizer_args=optimizer_args,
            # learning rate
            lr=lr,
            # schedulers
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            scheduler_args=scheduler_args,
            scheduler_optimizer=scheduler_optimizer,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            # instanciation configs
            scheduler_monitor=scheduler_monitor,
            save_hparams=save_hparams,
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
