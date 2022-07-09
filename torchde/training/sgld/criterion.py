import typing as th
import torch
import pytorch_lightning as pl
from torchde.training.criterion import Criterion, ResultsDict, FactorsDict
from torchde.training.terms import TermDescriptor


class SGLDTrainingCriterion(Criterion):
    """Training objective for Stochastic Gradient Langevin Dynamics (SGLD)"""

    def __init__(
        self,
        terms: th.List[TermDescriptor] = ("torchde.training.sgld.terms.SGLDContrastiveDivergenceTerm",),
        regularizations: th.Optional[th.List[TermDescriptor]] = (
            "torchde.training.sgld.terms.SGLDScoreRegularizationTerm",
        ),
        **kwargs,
    ):
        super().__init__(terms=terms, regularizations=regularizations, **kwargs)

    def __call__(
        self,
        *args,
        inputs: th.Any = None,
        samples: torch.Tensor = None,
        training_module: pl.LightningModule = None,
        return_factors: bool = True,
        **kwargs,
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        all_inputs = torch.cat([inputs, samples], dim=0)
        inputs_out, samples_out = training_module.model(all_inputs).chunk(2, dim=0)
        return super().__call__(
            *args,
            inputs=inputs,
            samples=samples,
            inputs_out=inputs_out,
            samples_out=samples_out,
            training_module=training_module,
            return_factors=return_factors,
            **kwargs,
        )
