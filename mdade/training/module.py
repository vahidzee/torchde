import torch
import typing as th
import functools
import pytorch_lightning as pl
from mdade.made import MADE
from mdade.utils import get_value, process_function_description
from .criterion import MADETrainingCriterion
from .attack import PGDAttacker


class MADETrainer(pl.LightningModule):
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
        model_cls: th.Optional[str] = "mdade.MADE",
        model_args: th.Optional[dict] = None,
        criterion_args: th.Optional[dict] = None,
        attack_args: th.Optional[dict] = None,
        inputs_transform: th.Optional[str] = None,
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
        self.model = get_value(self.hparams.model_cls)(**(self.hparams.model_args or dict()))

        # criterion and attacks can be different from the checkpointed model
        self.criterion = MADETrainingCriterion(
            {**(self.hparams.criterion_args or dict()), **(criterion_args or dict())}
        )
        self.attacker = (
            PGDAttacker(
                criterion=self.criterion, **{**(self.hparams.attack_args or dict()), **(attack_args or dict())}
            )
            if (self.hparams.attack_args or attack_args)
            else None
        )
        self.inputs_transform = inputs_transform

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
        self.model(inputs)

    def process_inputs(self, batch):
        "Process the inputs before forward pass"
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = self.inputs_transform_fucntion(inputs) if self.inputs_transform_fucntion else inputs
        if isinstance(self.model, MADE):
            # for mlp models
            inputs = inputs.reshape(inputs.shape[0], -1)

        return inputs

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
        inputs = self.process_inputs(batch)
        
        torch.set_grad_enabled(not is_val)
        if self.attacker and not is_val:
            adv_inputs, init_loss, final_loss = self.attacker(model=self.model, inputs=inputs, return_loss=True)
            results = self.criterion(model=self.model, inputs=adv_inputs)
            results["adv/init_loss"] = init_loss
            results["adv/final_loss"] = final_loss
            results["adv/loss_diff"] = final_loss - init_loss
        else:
            results = self.criterion(model=self.model, inputs=inputs)
        for item, value in results.items():
            self.log(
                f"{item}/{name}",
                value.mean(),
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
            )
        return results["loss"] if not is_val else None

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        "Pytorch Lightning's training_step function"
        return self.step(batch, batch_idx, optimizer_idx, name="train")

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        "Pytorch Lightning's validation_step function"
        return self.step(batch, batch_idx, optimizer_idx, name="val")
