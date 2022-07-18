import typing as th
import torch
import pytorch_lightning as pl
from torchde.training.criterion import Criterion, ResultsDict, FactorsDict
from torchde.training.terms import TermDescriptor


class EDETrainingCriterion(Criterion):
    """Training objective for Encoding Density Estimation Models"""

    def __init__(
        self,
        terms: th.Union[th.List[TermDescriptor], th.Tuple[TermDescriptor]] = (
            "torchde.training.encoding.terms.EDEStepTerm",
        ),
        regularizations: th.Optional[th.Union[th.List[TermDescriptor], th.Tuple[TermDescriptor]]] = None,
        **kwargs,
    ):
        super().__init__(terms=terms, regularizations=regularizations, **kwargs)

    def __call__(
        self,
        *args,
        inputs: th.Any = None,
        encodings: th.Optional[torch.Tensor] = None,
        training_module: pl.LightningModule = None,
        return_factors: bool = True,
        **kwargs,
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        encodings = encodings if encodings is not None else training_module.encoder(inputs)
        return super().__call__(
            *args,
            inputs=inputs,
            encodings=encodings,
            training_module=training_module,
            return_factors=return_factors,
            **kwargs,
        )
