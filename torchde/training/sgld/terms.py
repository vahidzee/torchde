import typing as th
import torch
import pytorch_lightning as pl
from torchde.training.terms import CriterionTerm


class SGLDContrastiveDivergenceTerm(CriterionTerm):
    def __init__(self, name: th.Optional[str] = None, **kwargs) -> None:
        super().__init__(name=name or "cdiv", **kwargs)

    def __call__(
        self,
        inputs,
        samples,
        training_module: pl.LightningModule,
        inputs_out: th.Optional[torch.Tensor] = None,
        samples_out: th.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if samples_out is not None and inputs_out is not None:
            return samples_out.mean() - inputs_out.mean()
        all_inputs = torch.cat([inputs, samples], dim=0)
        inputs_out, samples_out = training_module.model(all_inputs).chunk(2, dim=0)
        return samples_out.mean() - inputs_out.mean()


class SGLDScoreRegularizationTerm(CriterionTerm):
    """Centers the energy scores around 0. and prevents them from fluctuating too much."""

    def __init__(self, name: th.Optional[str] = None, factor=0.1, **kwargs) -> None:
        super().__init__(name=name or "score_regularization", factor=factor, **kwargs)

    def __call__(
        self,
        inputs,
        samples,
        training_module: pl.LightningModule,
        inputs_out: th.Optional[torch.Tensor] = None,
        samples_out: th.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if samples_out is not None and inputs_out is not None:
            return (samples_out**2 + inputs_out**2).mean()
        all_inputs = torch.cat([inputs, samples], dim=0)
        inputs_out, samples_out = training_module.model(all_inputs).chunk(2, dim=0)
        return (samples_out**2 + inputs_out**2).mean()
