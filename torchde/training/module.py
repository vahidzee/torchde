import torch
import typing as th
import functools
import pytorch_lightning as pl
from torchde.models import MADE
from torchde.utils import get_value, process_function_description, safe_function_call_wrapper, FunctionDescriptor
import torchmetrics
from .criterion import Criterion, ResultsDict
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
        anomaly_detection_score: th.Optional[th.Union[str, FunctionDescriptor]] = None,
        # criterion
        criterion: th.Union[Criterion, str] = "torchde.training.criterion.Criterion",
        criterion_args: th.Optional[dict] = None,
        # attacks
        attack_args: th.Optional[dict] = None,
        # input transforms
        inputs_transform: th.Optional[FunctionDescriptor] = None,
        inputs_noise_eps: th.Optional[float] = None,
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
        self.anomaly_detection_score_description = anomaly_detection_score
        if self.anomaly_detection_score_description is not None:
            # anomaly detection metrics
            self.val_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)
            self.test_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)

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
        self.inputs_transform = inputs_transform if inputs_transform is not None else self.hparams.inputs_transform
        self.lr = lr if lr is not None else self.hparams.lr

    @functools.cached_property
    def anomaly_detection_score(self):
        if self.anomaly_detection_score_description is None:
            return None
        if self.anomaly_detection_score_description in self.criterion.terms_names:
            function = lambda criterion_results: criterion_results[self.anomaly_detection_score_description]
        else:
            function = process_function_description(self.anomaly_detection_score_description, entry_function="score")
        return safe_function_call_wrapper(function)

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

    def log_step_results(self, results, factors, name: str = "train"):
        "Log the results of the step"
        is_val = name == "val"
        # logging results
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
        # validation step only logs
        if not is_val:
            return

        # logging factors
        for item, value in factors.items():
            self.log(
                f"factors/{item}/{name}",
                value.mean() if isinstance(value, torch.Tensor) else value,
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
            )

    def anomaly_detection_step(self, inputs, labels, results: ResultsDict, name: str = "val", **kwargs):
        is_val = name == "val"
        if not is_val or self.anomaly_detection_score is None:
            return results
        scores = self.anomaly_detection_score(criterion_results=results, training_module=self, inputs=inputs, **kwargs)
        self.val_auroc(
            preds=scores.reshape(-1),  # auroc expects predictions to have higher values for the positive class
            target=labels.reshape(-1),
        )
        normal_scores = scores[labels == 1]
        anomaly_scores = scores[labels == 0]
        if normal_scores.shape[0] != 0:
            results["anomaly_detection/score/normal"] = normal_scores.mean()
        if anomaly_scores.shape[0] != 0:
            results["anomaly_detection/score/anomaly"] = anomaly_scores.mean()
        results["metrics/auroc"] = self.val_auroc
        return results

    def step(
        self,
        batch,
        batch_idx: th.Optional[int] = None,
        optimizer_idx: th.Optional[int] = None,
        name: str = "train",
        inputs: th.Optional[th.Any] = None,
        labels: th.Optional[th.Any] = None,
        **kwargs,  # additional arguments to pass to the criterion and attacker
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
        if inputs is None:
            inputs, labels = self.process_inputs(batch)

        if self.attacker and not is_val:
            adv_inputs, init_loss, final_loss = self.attacker(
                model=self.model, training_module=self, inputs=inputs, return_loss=True, **kwargs
            )

        results, factors = self.criterion(
            inputs=adv_inputs if self.attacker and not is_val else inputs,
            training_module=self,
            return_factors=True,
            **kwargs,
        )

        # evaluate model's anomaly detection performance (only functional in validation steps)
        results = self.anomaly_detection_step(inputs, labels, results, name)

        # preparing results to be logged
        if self.attacker and not is_val:
            results["adversarial_attack/loss/initial"] = init_loss
            results["adversarial_attack/loss/final"] = final_loss
            results["adversarial_attack/loss/difference"] = final_loss - init_loss
        self.log_step_results(results, factors, name)
        return results["loss"] if not is_val else None

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
