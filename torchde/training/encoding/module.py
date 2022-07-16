import torch
import typing as th
import pytorch_lightning as pl
from torchde.training.module import DETrainingModule
from torchde.training.criterion import Criterion
from torchde.training.module import DETrainingModule
import torchde.utils


class EDETrainingModule(DETrainingModule):
    def __init__(
        self,
        # model
        encoder_model: th.Optional[torch.nn.Module] = None,
        encoder_model_cls: th.Optional[str] = None,
        encoder_model_args: th.Optional[dict] = None,
        de_model: th.Optional[DETrainingModule] = None,
        de_model_cls: th.Optional[str] = None,
        de_model_args: th.Optional[dict] = None,
        anomaly_detection_score: th.Optional[th.Union[str, torchde.utils.FunctionDescriptor]] = None,
        # criterion
        criterion: th.Optional[th.Union[Criterion, str]] = "torchde.training.encoding.criterion.EDETrainingCriterion",
        criterion_args: th.Optional[dict] = None,
        # attacks
        attack_args: th.Optional[dict] = None,
        # input transforms
        inputs_transform: th.Optional[torchde.utils.FunctionDescriptor] = None,
        inputs_noise_eps: th.Optional[float] = None,
        labels_transform: th.Optional[torchde.utils.FunctionDescriptor] = None,
        # optimization configs
        optimizer: str = "torch.optim.Adam",
        optimizer_parameters: th.Optional[th.Union[th.List[str], str]] = ("encoder", "density_estimator"),
        optimizer_args: th.Optional[dict] = None,
        # learning rate
        lr: th.Union[th.List[float], float] = 1e-4,
        # schedulers
        scheduler: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_optimizer: th.Optional[th.Union[int, th.List[int]]] = None,
        scheduler_args: th.Optional[th.Union[dict, th.List[dict]]] = None,
        scheduler_interval: th.Union[str, th.List[str]] = "epoch",
        scheduler_frequency: th.Union[int, th.List[int]] = 1,
        scheduler_monitor: th.Optional[th.Union[str, th.List[str]]] = None,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["encoder_model", "de_model"])
        super().__init__(
            # model
            model=None,
            model_cls=None,
            model_args=None,
            # anomaly detection score
            anomaly_detection_score=anomaly_detection_score,
            # criterion
            criterion=criterion,
            criterion_args=criterion_args,
            # attacks
            attack_args=attack_args,
            # input transforms
            inputs_transform=inputs_transform,
            inputs_noise_eps=inputs_noise_eps,
            labels_transform=labels_transform,
            # optimization configs
            optimizer=optimizer,
            optimizer_parameters=optimizer_parameters,
            optimizer_args=optimizer_args,
            # learning rate
            lr=lr,
            # schedulers
            scheduler=scheduler,
            scheduler_optimizer=scheduler_optimizer,
            scheduler_args=scheduler_args,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_monitor=scheduler_monitor,
            # instantiation configurations
            save_hparams=False,
        )
        # instanciate encoder
        self.encoder = (
            encoder_model
            if encoder_model is not None
            else torchde.utils.get_value(self.hparams.encoder_model_cls)(**(self.hparams.encoder_model_args or dict()))
        )
        # instanciate density estimator model
        self.density_estimator = (
            de_model
            if de_model is not None
            else torchde.utils.get_value(self.hparams.de_model_cls)(**(self.hparams.de_model_args or dict()))
        )
        assert isinstance(
            self.density_estimator, DETrainingModule
        )  # main training criterion comes from the density estimator

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    # def training_step(self, batch, batch_idx, optimizer_idx=None):
    #     "Pytorch Lightning's training_step function"

    #     loss = self.step(batch, batch_idx, optimizer_idx, name="train")
    #     return loss
