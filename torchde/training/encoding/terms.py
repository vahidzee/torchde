import typing as th
import torch
import pytorch_lightning as pl
from torchde.training.terms import CriterionTerm


class EDEStepTerm(CriterionTerm):
    def __init__(self, name: th.Optional[str] = None, **kwargs) -> None:
        super().__init__(name=name or "density_estimator", **kwargs)

    def __call__(
        self,
        inputs,
        training_module: pl.LightningModule,
        labels: th.Optional[th.Any] = None,
        encodings: th.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encodings = encodings if encodings is not None else training_module.encoder(inputs)
        step_results = training_module.density_estimator.step(
            inputs=encodings,
            labels=labels,
            return_results=True,
            return_factors=False,
            name="train",
            log_results=False,
            transform_inputs=True,
            transform_labels=True,
        )
        return step_results
