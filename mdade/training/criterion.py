import typing as th
import torch
import functools
from mdade.utils import process_function_description, safe_function_call_wrapper
from mdade.made import MADE

# types
FactorsDict = th.Dict[str, torch.Tensor]
ResultsDict = th.Dict[str, torch.Tensor]


class MADETrainingCriterion:
    """Training objective for Masked Autoregressive Models.

    Attributes:
        params_regularizations (dict):
            A dictionary of regularization functions for the parameters of the model.
        params_regularizations_factors (dict):
            A dictionary of factors for the regularization functions for the parameters of the model.
        scale_regularizations (bool):
            Whether to scale the regularization losses to be in the same range as the negative log-likelihood.
    """

    def __init__(
        self,
        params_regularizations: th.Optional[th.Dict[str, str]] = None,
        params_regularizations_factors: th.Optional[th.Dict[str, str]] = None,
        scale_regularizations: bool = False,
    ):
        self.params_regularizations = params_regularizations or dict()
        self.params_regularizations_factors = params_regularizations_factors or dict()
        self.scale_regularizations = scale_regularizations

    @functools.cached_property
    def params_regularization_factor_functions(self):
        return {
            name: safe_function_call_wrapper(process_function_description(value, entry_function="factor"))
            for name, value in self.params_regularizations_factors.items()
        }

    @functools.cached_property
    def params_regularization_functions(self):
        return {
            name: safe_function_call_wrapper(process_function_description(value, entry_function="regularize"))
            for name, value in self.params_regularizations.items()
        }

    def __call__(
        self, model: MADE, inputs, trainer=None, return_factors=False
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        if not self.params_regularization_functions:
            # shortcut for trainings with no parameter regularizations
            nll = -model.log_prob(inputs, reduce=True).mean()
            return dict(loss=nll, nll=nll) if not return_factors else (dict(loss=nll, nll=nll), dict())

        results = dict(nll=0.0, loss=0.0)
        regularizations = {name: 0.0 for name in self.params_regularization}
        for i in range(model.num_masks or 1):
            model.reorder()
            params_logits = model(inputs)
            params = model.density_estimator.transform_distribution_parameters(params_logits)
            for name, func in self.params_regularization_functions.items():
                regularizations[name] += func(
                    params[name], trainer=trainer, model=model
                ).mean()  # mean will ensure that regularization results are scalars
            results["nll"] -= model.density_estimator.log_prob(
                inputs,
                params_logits=params_logits,
                params=params,
                reduce=True,
            ).mean()

        # normalizing by the number of masks
        factors = dict()
        results["nll"] /= model.num_masks or 1
        results["loss"] += results["nll"]
        for name in regularizations:
            regularizations[name] /= model.num_masks or 1
            results[f"params_regularization/{name}"] = regularizations[name]

            factor = self.params_regularization_factor.get(name, 1.0)
            factors[name] = factor(regularizations[name], trainer=trainer, model=model) if callable(factor) else factor

            results["loss"] += (
                (regularizations[name] / regularizations[name].data.abs() * results["nll"].data)
                if self.scale_regularizations
                else regularizations[name]
            ) * factors[name]
        return results if not return_factors else results, factors

    def __repr__(self):
        regularizations = f"params_regularization={self.params_regularizations}" if self.params_regularizations else ""
        regularization_factors = (
            f", params_regularization_factor={self.params_regularizations_factors}"
            if self.params_regularizations_factors
            else ""
        )
        return f"{self.__class__.__name__}({regularizations}{regularization_factors})"
