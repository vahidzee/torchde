import torch
import typing as th
import functools
import pytorch_lightning as pl
import torchde.utils
import torchde.training.utils
from torchde.utils import FunctionDescriptor, process_function_description, safe_function_call_wrapper
from .criterion import Criterion, ResultsDict
from .attack import PGDAttacker
import torchmetrics
import torch
import types


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
        # anomaly detection [score(criterion_results, training_module, inputs) -> torch.Tensor]
        anomaly_detection_score: th.Optional[th.Union[str, FunctionDescriptor]] = None,
        # criterion
        criterion: th.Union[Criterion, str] = "torchde.training.Criterion",
        criterion_args: th.Optional[dict] = None,
        # attacks
        attack_args: th.Optional[dict] = None,
        # input transforms [transform(inputs) -> torch.Tensor]
        inputs_transform: th.Optional[FunctionDescriptor] = None,
        inputs_noise_eps: th.Optional[float] = None,
        labels_transform: th.Optional[FunctionDescriptor] = None,
        # optimization configs [is_active(training_module, optimizer_idx) -> bool]
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
        # initialization settings
        save_hparams: bool = True,
        initialize_superclass: bool = True,
    ) -> None:
        """Initialize the trainer.

        Args:
            model_cls: the class of the model to use (import path)
            model_args: the arguments to pass to the model constructor
            criterion_args: the arguments to pass to the criterion constructor
            attack_args: the arguments to pass to the attacker constructor (PGDAttacker)
            inputs_transform:
                the transform function to apply to the inputs before forward pass, can be used for
                applying dequantizations.

        Returns:
            None
        """
        if initialize_superclass:
            super().__init__()
        if save_hparams:
            self.save_hyperparameters(ignore=["model"])
        # criterion and attacks can be different from the checkpointed model
        criterion = criterion if criterion is not None else self.hparams.criterion
        criterion = torchde.utils.get_value(criterion) if isinstance(criterion, str) else criterion
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

        self.anomaly_detection_score_description = (
            anomaly_detection_score if anomaly_detection_score is not None else self.hparams.anomaly_detection_score
        )
        self.inputs_transform = inputs_transform if inputs_transform is not None else self.hparams.inputs_transform
        self.inputs_noise_eps = inputs_noise_eps if inputs_noise_eps is not None else self.hparams.inputs_noise_eps
        self.labels_transform = labels_transform if labels_transform is not None else self.hparams.labels_transform

        if self.anomaly_detection_score_description is not None:
            # anomaly detection metrics
            self.val_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)
            self.test_auroc = torchmetrics.AUROC(num_classes=2, pos_label=1)

        # optimizers and schedulers
        if (optimizer if optimizer is not None else self.hparams.optimizer) is not None:
            # optimizers
            (
                self.optimizer,
                self.optimizer_is_active_descriptor,
                self.optimizer_parameters,
                self.optimizer_args,
            ), optimizers_count = torchde.utils.list_args(
                optimizer if optimizer is not None else self.hparams.optimizer,
                optimizer_is_active if optimizer_is_active is not None else self.hparams.optimizer_is_active,
                optimizer_parameters if optimizer_parameters is not None else self.hparams.optimizer_parameters,
                optimizer_args if optimizer_args is not None else self.hparams.optimizer_args,
                return_length=True,
            )
            self.optimizer_is_active_descriptor = (
                None
                if all(i is None for i in self.optimizer_is_active_descriptor)
                else self.optimizer_is_active_descriptor
            )

            # learning rates
            self.lr = torchde.utils.list_args(lr if lr is not None else self.hparams.lr, length=optimizers_count)

            (
                (
                    self.scheduler,
                    self.scheduler_name,
                    self.scheduler_optimizer,
                    self.scheduler_args,
                    self.scheduler_interval,
                    self.scheduler_frequency,
                    self.scheduler_monitor,
                ),
                schedulers_count,
            ) = torchde.utils.list_args(
                scheduler if scheduler is not None else self.hparams.scheduler,
                scheduler_name if scheduler_name is not None else self.hparams.scheduler_name,
                scheduler_optimizer if scheduler_optimizer is not None else self.hparams.scheduler_optimizer,
                scheduler_args if scheduler_args is not None else self.hparams.scheduler_args,
                scheduler_interval if scheduler_interval is not None else self.hparams.scheduler_interval,
                scheduler_frequency if scheduler_frequency is not None else self.hparams.scheduler_frequency,
                scheduler_monitor if scheduler_monitor is not None else self.hparams.scheduler_monitor,
                return_length=True,
            )
            schedulers_count = schedulers_count if schedulers_count and self.scheduler[0] is not None else 0
            if not schedulers_count:
                (
                    self.scheduler,
                    self.scheduler_name,
                    self.scheduler_optimizer,
                    self.scheduler_args,
                    self.scheduler_frequency,
                    self.scheduler_interval,
                    self.scheduler_monitor,
                ) = (None, None, None, None, None, None, None)
            if (schedulers_count == 1 and optimizers_count > 1) or (
                schedulers_count > 0 and all(self.scheduler_optimizer[i] is None for i in range(schedulers_count))
            ):
                self.scheduler_optimizer = [i for j in range(schedulers_count) for i in range(optimizers_count)]
                self.scheduler = [j for j in self.scheduler for i in range(optimizers_count)]
                self.scheduler_name = [j for j in self.scheduler_name for i in range(optimizers_count)]
                self.scheduler_args = [j for j in self.scheduler_args for i in range(optimizers_count)]
                self.scheduler_interval = [j for j in self.scheduler_interval for i in range(optimizers_count)]
                self.scheduler_frequency = [j for j in self.scheduler_frequency for i in range(optimizers_count)]
                self.scheduler_monitor = [j for j in self.scheduler_monitor for i in range(optimizers_count)]
            if schedulers_count:
                for idx, name in enumerate(self.scheduler_name):
                    param_name = self.optimizer_parameters[self.scheduler_optimizer[idx]]
                    param_name = (
                        f"/{param_name}"
                        if param_name and isinstance(param_name, str)
                        else f"/{self.scheduler_optimizer[idx]}"
                    )
                    self.scheduler_name[idx] = f"lr_scheduler{param_name}/{name if name is not None else idx}"
                self.__scheduler_step_count = [0 for i in range(len(self.scheduler))]

        if hasattr(self, "optimizer_is_active_descriptor") and self.optimizer_is_active_descriptor is not None:
            self.automatic_optimization = False
            self.training_step = types.MethodType(DETrainingModule.training_step_manual, self)
            self.__params_frozen = [False for i in range(optimizers_count)]
            self.__params_state = [None for i in range(optimizers_count)]
        else:
            self.training_step = types.MethodType(DETrainingModule.training_step_automatic, self)

        # initialize the model
        if (
            model is not None
            or model_args is not None
            or model_cls is not None
            or (hasattr(self.hparams, "model_cls") and self.hparams.model_cls is not None)
            or (hasattr(self.hparams, "model_args") and self.hparams.model_args is not None)
        ):
            self.model = (
                model
                if model is not None
                else torchde.utils.get_value(self.hparams.model_cls)(**(self.hparams.model_args or dict()))
            )

    @functools.cached_property
    def optimizer_is_active(self):
        if self.optimizer_is_active_descriptor is None:
            return None
        return [
            safe_function_call_wrapper(process_function_description(i, entry_function="is_active"))
            for i in self.optimizer_is_active_descriptor
        ]

    def forward(self, inputs):
        "Placeholder forward pass for the model"
        if hasattr(self, "model") and self.model is not None:
            return self.model(inputs)
        raise NotImplementedError("No model defined")

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
        inputs, labels = self.process_inputs(
            batch, inputs=inputs, labels=labels, transform_inputs=transform_inputs, transform_labels=transform_labels
        )
        if self.attacker and not is_val:
            adv_inputs, init_loss, final_loss = self.attacker(
                training_module=self, inputs=inputs, return_loss=True, **kwargs
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
        if log_results:
            self.log_step_results(results, factors, name)
        if return_results:
            return (results, factors) if return_factors else results
        return results["loss"] if not is_val else None

    def configure_optimizers(self):
        optimizers = [
            self.__configure_optimizer(
                opt_class=opt_cls, opt_args=opt_args, opt_base_lr=opt_base_lr, opt_parameters=opt_parameters
            )
            for opt_cls, opt_base_lr, opt_parameters, opt_args in zip(
                self.optimizer,
                self.lr,
                self.optimizer_parameters,
                self.optimizer_args,
            )
        ]
        schedulers = (
            [
                self.__configure_scheduler(
                    sched_class=sched_cls,
                    sched_optimizer=optimizers[sched_optimizer if sched_optimizer is not None else 0],
                    sched_args=sched_args,
                    sched_interval=sched_interval,
                    sched_frequency=sched_frequency,
                    sched_monitor=sched_monitor,
                    sched_name=sched_name,
                )
                for sched_cls, sched_optimizer, sched_args, sched_interval, sched_frequency, sched_monitor, sched_name in zip(
                    self.scheduler,
                    self.scheduler_optimizer,
                    self.scheduler_args,
                    self.scheduler_interval,
                    self.scheduler_frequency,
                    self.scheduler_monitor,
                    self.scheduler_name,
                )
            ]
            if self.scheduler
            else None
        )
        if schedulers:
            return (
                dict(optimizer=optimizers[0], scheduler=schedulers[0])
                if len(schedulers) == 1 and len(optimizers) == 1
                else (
                    optimizers,
                    schedulers,
                )
            )
        return optimizers

    def __configure_optimizer(self, opt_class, opt_base_lr, opt_args, opt_parameters):
        opt_class = torchde.utils.get_value(opt_class)
        opt_args = {"lr": opt_base_lr, **(opt_args if opt_args is not None else {})}
        opt = opt_class(
            (torchde.utils.get_value(opt_parameters, context=self) if opt_parameters else self).parameters(),
            **opt_args,
        )
        return opt

    def __configure_scheduler(
        self, sched_class, sched_optimizer, sched_args, sched_interval, sched_frequency, sched_monitor, sched_name
    ):
        sched_class = torchde.utils.get_value(sched_class)
        sched_args = sched_args or dict()
        sched = sched_class(sched_optimizer, **sched_args)
        sched_instance_dict = dict(
            scheduler=sched,
            interval=sched_interval,
            frequency=sched_frequency,
            name=sched_name,
            reduce_on_plateau=isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau),
        )
        if sched_monitor is not None:
            sched_instance_dict["monitor"] = sched_monitor
        return sched_instance_dict

    @functools.cached_property
    def anomaly_detection_score(self):
        if self.anomaly_detection_score_description is None:
            return None
        if self.anomaly_detection_score_description in self.criterion.terms_names:
            function = lambda criterion_results: criterion_results[self.anomaly_detection_score_description]
        else:
            function = torchde.utils.process_function_description(
                self.anomaly_detection_score_description, entry_function="score"
            )
        return torchde.utils.safe_function_call_wrapper(function)

    @functools.cached_property
    def inputs_transform_fucntion(self):
        """The transform function (callable) to apply to the inputs before forward pass.

        Returns:
            The compiled callable transform function if `self.inputs_transform` is provided else None
        """
        return (
            torchde.utils.process_function_description(self.inputs_transform, entry_function="transform")
            if self.inputs_transform
            else None
        )

    @functools.cached_property
    def labels_transform_function(self):
        """The transform function (callable) to apply to the labels before forward pass.

        Returns:
            The compiled callable transform function if `self.labels_transform` is provided else None
        """
        return (
            torchde.utils.process_function_description(self.labels_transform, entry_function="transform")
            if self.labels_transform
            else None
        )

    def process_inputs(
        self,
        batch,
        inputs: th.Optional[th.Any] = None,
        labels: th.Optional[th.Any] = None,
        transform_inputs: bool = True,
        transform_labels: bool = True,
    ):
        "Process the inputs before forward pass"
        inputs = inputs if inputs is not None else (batch[0] if isinstance(batch, (tuple, list)) else batch)
        labels = labels if labels is not None else (batch[1] if isinstance(batch, (tuple, list)) else None)
        if self.inputs_noise_eps:
            inputs = inputs + torch.randn_like(inputs) * self.inputs_noise_eps
        if transform_inputs:
            inputs = (
                self.inputs_transform_fucntion(inputs, training_module=self)
                if self.inputs_transform_fucntion
                else inputs
            )
        if transform_labels:
            labels = (
                self.labels_transform_function(labels, training_module=self)
                if self.labels_transform_function
                else labels
            )
        if batch is None:
            return inputs, labels if labels is not None else inputs
        return inputs, labels if isinstance(batch, (tuple, list)) else inputs

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
        if anomaly_scores.shape[0] != 0 and normal_scores.shape[0] != 0:
            results["anomaly_detection/score/difference"] = normal_scores.mean() - anomaly_scores.mean()
        results["metrics/auroc"] = self.val_auroc
        return results

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

    def training_step_automatic(self, batch, batch_idx, optimizer_idx=None, **kwargs):
        "Implementation for automatic Pytorch Lightning's training_step function"
        return self.step(batch, batch_idx, optimizer_idx, name="train", **kwargs)

    def manual_lr_schedulers_step(self, scheduler, scheduler_idx, **kwargs):
        "Implementation for manual Pytorch Lightning's lr_step function"
        frequency = self.scheduler_frequency[scheduler_idx]
        if not frequency:
            return
        interval = self.scheduler_interval[scheduler_idx]
        monitor = self.scheduler_monitor[scheduler_idx]

        step = False
        if interval == "batch":
            self.__scheduler_step_count[scheduler_idx] = (1 + self.__scheduler_step_count[scheduler_idx]) % frequency
            step = not self.__scheduler_step_count[scheduler_idx]
        elif interval == "epoch":
            step = self.trainer.is_last_batch and not (self.trainer.current_epoch % frequency)
        if not step:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if monitor not in self.trainer.callback_metrics:
                return  # no metric to monitor, skip scheduler step until metric is available in next loops
            scheduler.step(self.trainer.callback_metrics[monitor])
        else:
            scheduler.step()

    def training_step_manual(self, batch, batch_idx, **kwargs):
        "Implementation for manual training and optimization"
        optimizers = self.optimizers()
        optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        schedulers = self.lr_schedulers()
        schedulers = schedulers if isinstance(schedulers, (list, tuple)) else ([schedulers] if schedulers else [])
        optimizer_is_active = [
            self.optimizer_is_active[i](training_module=self, optimizer_idx=i) for i in range(len(optimizers))
        ]
        # freezing/unfreezing the optimizer parameters
        for optimizer_idx, optimizer in enumerate(optimizers):
            if optimizer_is_active[optimizer_idx] and self.__params_frozen[optimizer_idx]:
                torchde.training.utils.unfreeze_params(
                    optimizer=optimizer, old_states=self.__params_state[optimizer_idx]
                )
                self.__params_frozen[optimizer_idx] = False
            elif not optimizer_is_active[optimizer_idx] and not self.__params_frozen[optimizer_idx]:
                self.__params_state[optimizer_idx] = torchde.training.utils.freeze_params(optimizer=optimizer)
                self.__params_frozen[optimizer_idx] = True
        loss = self.step(batch, batch_idx, None, name="train")
        self.manual_backward(loss)
        for optimizer_idx, optimizer in enumerate(optimizers):
            if optimizer_is_active[optimizer_idx]:
                optimizer.step()  # todo: add support for LBFGS optimizers via closures
                optimizer.zero_grad()  # todo: move to before the backward call and add support for gradient accumulation
        # following pytorch>=1.1.0 conventions, calling scheduler.step after optimizer.step
        # visit the docs for more details https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        for idx, scheduler in enumerate(schedulers):
            if optimizer_is_active[self.scheduler_optimizer[idx]]:
                self.manual_lr_schedulers_step(scheduler=scheduler, scheduler_idx=idx)
        return loss

    def validation_step(self, batch, batch_idx):
        "Pytorch Lightning's validation_step function"
        return self.step(batch, batch_idx, name="val")
