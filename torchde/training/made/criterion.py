import typing as th
import pytorch_lightning as pl
from torchde.training.criterion import Criterion, ResultsDict, FactorsDict
from torchde.training.terms import CriterionTerm, TermDescriptor


class MADETrainingCriterion(Criterion):
    """Training objective for Masked Autoregressive Models.

    Attributes:
        terms: List of terms to use in the training objective.

    """

    def __init__(
        self,
        terms: th.List[TermDescriptor] = ("torchde.training.made.terms.MADENLLTerm",),
        params_regularizations: th.Optional[th.List[TermDescriptor]] = None,
        regularizations: th.Optional[th.List[TermDescriptor]] = None,
        # reductions
        terms_reduction: str = "sum",  # sum or multiply
        regularizations_reduction: str = "sum",  # sum or multiply
        overall_reduction: str = "sum",  # sum or multiply
    ):
        """Initialize the training criterion."""
        super().__init__(
            terms=terms,
            regularizations=regularizations,
            terms_reduction=terms_reduction,
            regularizations_reduction=regularizations_reduction,
            overall_reduction=overall_reduction,
        )
        self.params_regularizations = [CriterionTerm.from_description(term) for term in (params_regularizations or [])]
        self.rename_terms(self.params_regularizations, "regularization/params/")
        self.regularizations += self.params_regularizations

    def __call__(
        self,
        *args,
        inputs: th.Any = None,
        training_module: pl.LightningModule = None,
        return_factors: bool = True,
        **kwargs,
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        if not self.params_regularizations:
            return super().__call__(
                *args, inputs=inputs, training_module=training_module, return_factors=return_factors, **kwargs
            )

        # compute term results and regularizations for all masks
        results = dict()
        for i in range(training_module.model.num_masks or 1):
            training_module.model.reorder()
            params_logits = training_module.model(inputs)
            params = training_module.model.density_estimator.transform_distribution_parameters(params_logits)

            results = {
                term_name: value + results.get(term_name, 0.0)
                for term_name, value in self.process_term_results(
                    training_module=training_module,
                    inputs=inputs,
                    params=params,
                    params_logits=params_logits,
                    terms=self.terms + self.regularizations,
                ).items()
            }

        # normalize by the number of masks
        results = {name: value / (training_module.model.num_masks or 1) for name, value in results.items()}

        # compute factors and reduce results (similar to base class)
        factors = {
            term.name: term.factor_value(results_dict=results, training_module=training_module) for term in self.terms
        }
        results["loss"] = self.reduce(results, results, terms_name="terms")
        factors = {
            **factors,
            **{
                term.name: term.factor_value(results_dict=results, training_module=training_module)
                for term in self.regularizations
            },
        }
        regularizations_reduced = self.reduce(results, factors, terms_name="regularizations")
        results["loss"] = (
            (results["loss"] + regularizations_reduced)
            if self.overall_reduction == "sum"
            else (results["loss"] * regularizations_reduced)
        )
        return results if not return_factors else (results, factors)
