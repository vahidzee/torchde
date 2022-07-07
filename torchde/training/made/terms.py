import typing as th
import torch
import pytorch_lightning as pl
from torchde.training.terms import CriterionTerm


class MADENLLTerm(CriterionTerm):
    def __init__(self, name: th.Optional[str] = None, **kwargs) -> None:
        super().__init__(name=name or "nll", **kwargs)

    def __call__(
        self,
        inputs,
        training_module: pl.LightningModule,
        params_logits: th.Optional[torch.Tensor] = None,
        params: th.Optional[dict] = None,
    ):
        model = training_module.model
        if params_logits is None and params is None:
            return -model.log_prob(inputs, reduce=True).mean()
        return -model.density_estimator.log_prob(  # using density estimator's log-prob to bypass maskings
            inputs, params_logits=params_logits, params=params, reduce=True
        ).mean()


class MADEEntropyTerm(CriterionTerm):
    def __init__(self, name: th.Optional[str] = None, **kwargs) -> None:
        super().__init__(name=name or "entropy", **kwargs)

    def __call__(
        self,
        inputs,
        training_module: pl.LightningModule,
        params_logits: th.Optional[torch.Tensor] = None,
        params: th.Optional[dict] = None,
    ):
        model = training_module.model
        if params_logits is None and params is None:
            log_probs = model.log_prob(inputs, reduce=True)
        else:
            log_probs = model.density_estimator.log_prob(  # using density estimator's log-prob to bypass maskings
                inputs, params_logits=params_logits, params=params, reduce=True
            )
        return -(torch.exp(log_probs) * log_probs).sum()
