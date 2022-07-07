import typing as th
import torch
import pytorch_lightning as pl
import functools
from torchde.training.terms import CriterionTerm

# types
ResultsDict = th.Dict[str, torch.Tensor]
FactorsDict = th.Dict[str, torch.Tensor]


class Criterion:
    """Generic training objective abstraction

    Attributes:
        terms: List of terms to use in the training objective.
        regularizations: List of regularizations to use in the training objective.
        terms_reduction: Reduction method to use for terms.
        regularizations_reduction: Reduction method to use for regularizations.
        overall_reduction: Reduction method to use for overall loss.
    """

    def __init__(
        self,
        terms: th.List[th.Union[str, dict]],
        regularizations: th.Optional[th.List[th.Union[str, dict]]] = None,
        terms_reduction: str = "sum",  # sum or multiply
        regularizations_reduction: str = "sum",  # sum or multiply
        overall_reduction: str = "sum",  # sum or multiply
    ):
        self.terms = [
            CriterionTerm.from_description(
                term, factor_application="add" if terms_reduction == "multiply" else "multiply"
            )
            for term in terms
        ]
        self.regularizations = [
            CriterionTerm.from_description(
                term, factor_application="add" if regularizations_reduction == "multiply" else "multiply"
            )
            for term in (regularizations or [])
        ]
        self.rename_terms(terms=self.terms, prefix="term/")
        self.rename_terms(terms=self.regularizations, prefix="regularization/")
        self.terms_reduction = terms_reduction
        self.regularizations_reduction = regularizations_reduction
        self.overall_reduction = overall_reduction

    @staticmethod
    def rename_terms(terms: th.List[CriterionTerm], prefix: str = "") -> None:
        names_count = {term.name: 0 for term in terms}
        for term in terms:
            names_count[term.name] += 1
        for name in names_count:
            names_count[name] = names_count[name] if names_count[name] > 1 else -1
        for term in terms[::-1]:
            names_count[term.name] -= 1
            term.name = (
                f"{prefix}{term.name}"
                if names_count[term.name] < 0
                else f"{prefix}{term.name}/{names_count[term.name]}"
            )

    def process_term_results(
        self,
        *args,
        inputs: th.Any = None,
        training_module: pl.LightningModule = None,
        terms: th.Union[str, th.List[CriterionTerm]] = "terms",
        **kwargs,
    ):
        return {
            term.name: term(*args, inputs=inputs, training_module=training_module, **kwargs)
            for term in (getattr(self, terms) if isinstance(terms, str) else terms)
        }

    def reduce(self, term_results: ResultsDict, factors_dict: FactorsDict, terms_name: str = "terms") -> torch.Tensor:
        reduction = getattr(self, f"{terms_name}_reduction")
        factors_applied_values = [
            term.apply_factor(term_value=term_results[term.name], factor_value=factors_dict[term.name])
            for term in getattr(self, terms_name)
        ]
        if reduction == "sum":
            return sum(factors_applied_values)
        elif reduction == "multiply":
            return functools.reduce(lambda x, y: x * y, factors_applied_values)

    def __call__(
        self,
        *args,
        inputs: th.Any = None,
        training_module: pl.LightningModule = None,
        return_factors: bool = True,
        **kwargs,
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        results = self.process_term_results(
            *args, inputs=inputs, training_module=training_module, **kwargs, terms=self.terms + self.regularizations
        )
        factors = {
            term.name: term.factor_value(results_dict=results, training_module=training_module) for term in self.terms
        }
        results["loss"] = self.reduce(term_results=results, factors_dict=factors, terms_name="terms")
        factors = {
            **factors,
            **{
                term.name: term.factor_value(results_dict=results, training_module=training_module)
                for term in self.regularizations
            },
        }
        regularizations_reduced = self.reduce(term_results=results, factors_dict=factors, terms_name="regularizations")
        results["loss"] = (
            (results["loss"] + regularizations_reduced)
            if self.overall_reduction == "sum"
            else (results["loss"] * regularizations_reduced)
        )
        return results if not return_factors else (results, factors)
