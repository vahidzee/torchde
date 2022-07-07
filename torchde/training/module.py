import torch
import typing as th
import functools
import pytorch_lightning as pl
from torchde.models import MADE
from torchde.utils import get_value, process_function_description, FunctionDescriptor
import torchmetrics
from .criterion import Criterion
from .attack import PGDAttacker


class DETrainingModule(pl.LightningModule):
    """
    Generic Lightning Module for training MADE models.

    Attributes:
        model: the MADE model (import path)
        criterion: the criterion to use for training
        attack: the attack to use for training
        inputs_transform: the transform to apply to the inputs before forward pass
    """

    def __init__(
        self,
        # model
        model: th.Optional[torch.nn.Module] = None,
        model_cls: th.Optional[str] = None,
        model_args: th.Optional[dict] = None,
        anomaly_detector_score: th.Optional[th.Union[str, FunctionDescriptor]] = None,
        # criterion
        criterion: th.Union[Criterion, str] = "torchde.training.criterion.Criterion",
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
        """Initialize the trainer.

        Args:
            model_cls: the class of the model to use (import path)
            model_args: the arguments to pass to the model constructor
            criterion_args: the arguments to pass to the criterion constructor (MADETrainingCriterion)
            attack_args: the arguments to pass to the attacker constructor (PGDAttacker)
            inputs_transform:
                the transform function to apply to the inputs before forward pass, can be used for
                applying dequantizations.

        Returns:
            None
        """
        super().__init__()
        self.save_hyperparameters()
        # initialize the model
        self.model = (
            model if model is not None else get_value(self.hparams.model_cls)(**(self.hparams.model_args or dict()))
        )
        # anomaly detection

        # criterion and attacks can be different from the checkpointed model
        criterion = criterion if criterion is not None else self.hparams.criterion
        criterion = get_value(criterion) if isinstance(criterion, str) else criterion
        self.criterion = (
            criterion
            if isinstance(criterion, Criterion)
            else criterion(**{**(self.hparams.criterion_args or dict()), **(criterion_args or dict())})
        )

        self.attacker = (
            PGDAttacker(
                criterion=self.criterion, **{**(self.hparams.attack_args or dict()), **(attack_args or dict())}
            )
            if (self.hparams.attack_args or attack_args)
            else None
        )
        self.inputs_transform = inputs_transform
        self.lr = self.hparams.lr

        # metrics
        if False:
            self.val_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)
            self.test_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)

    @functools.cached_property
    def inputs_transform_fucntion(self):
        """The transform function (callable) to apply to the inputs before forward pass.

        Returns:
            The compiled callable transform function if `self.inputs_transform` is provided else None
        """
        return (
            process_function_description(self.inputs_transform, entry_function="transform")
            if self.inputs_transform
            else None
        )

    def forward(self, inputs):
        "Placeholder forward pass for the model"
        return self.model(inputs)

    def process_inputs(self, batch):
        "Process the inputs before forward pass"
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = self.inputs_transform_fucntion(inputs) if self.inputs_transform_fucntion else inputs
        if isinstance(self.model, MADE):
            # for mlp models
            inputs = inputs.reshape(inputs.shape[0], -1)

        return inputs, batch[1] if isinstance(batch, (tuple, list)) else None

    def step(
        self,
        batch,
        batch_idx: th.Optional[int] = None,
        optimizer_idx: th.Optional[int] = None,
        name: str = "train",
    ):
        """Train or evaluate the model with the given batch.

        Args:
            batch: batch of data to train or evaluate with
            batch_idx: index of the batch
            optimizer_idx: index of the optimizer
            name: name of the step ("train" or "val")

        Returns:
            None if the model is in evaluation mode, else a tensor with the training objective
        """
        is_val = name == "val"
        inputs, labels = self.process_inputs(batch)

        torch.set_grad_enabled(not is_val)
        if self.attacker and not is_val:
            adv_inputs, init_loss, final_loss = self.attacker(
                model=self.model, training_module=self, inputs=inputs, return_loss=True
            )
            results, factors = self.criterion(
                inputs=adv_inputs, training_module=self, return_factors=True
            )
            results["adversarial_attack/loss/initial"] = init_loss
            results["adversarial_attack/loss/final"] = final_loss
            results["adversarial_attack/loss/difference"] = final_loss - init_loss
        else:
            results, factors = self.criterion(inputs=inputs, training_module=self, return_factors=True)
        for item, value in results.items():
            self.log(
                f"{item}/{name}",
                value.mean() if isinstance(value, torch.Tensor) else value,
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
                prog_bar=is_val and name == "loss",
            )
        if is_val:
            return
        for item, value in factors.items():
            self.log(
                f"factors/{item}/{name}",
                value.mean() if isinstance(value, torch.Tensor) else value,
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
            )
        return results["loss"]

    def configure_optimizers(self):
        opt_class, opt_dict = torch.optim.Adam, {"lr": self.lr}
        if self.hparams.optimizer:
            opt_class = get_value(self.hparams.optimizer)
            opt_dict = self.hparams.optimizer_args or dict()
            opt_dict["lr"] = self.lr
        opt = opt_class(self.parameters(), **opt_dict)
        if self.hparams.scheduler is None:
            return opt
        sched_class = get_value(self.hparams.scheduler)
        sched_dict = self.hparams.scheduler_args or dict()
        sched = sched_class(opt, **sched_dict)
        sched_instance_dict = dict(
            scheduler=sched,
            interval=self.hparams.scheduler_interval,
            frequency=self.hparams.scheduler_frequency,
            reduce_on_plateau=isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau),
        )
        if self.hparams.scheduler_monitor:
            sched_instance_dict["monitor"] = self.hparams.scheduler_monitor
        return [opt], [sched_instance_dict]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        "Pytorch Lightning's training_step function"
        return self.step(batch, batch_idx, optimizer_idx, name="train")

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        "Pytorch Lightning's validation_step function"
        return self.step(batch, batch_idx, optimizer_idx, name="val")
