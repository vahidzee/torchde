import typing as th
import torch
from ..utils import process_function_description
from ..made import MADE

# types
FactorsDict = th.Dict[str, torch.Tensor]
ResultsDict = th.Dict[str, torch.Tensor]


class MADETrainingCriterion:
    def __init__(
        self,
        params_regularization: th.Optional[th.Dict[str, str]] = None,
        params_regularization_factor: th.Optional[th.Dict[str, str]] = None,
        scale_regularizations: bool = False,
    ):
        self.params_regularization = {
            name: process_function_description(value, entry_function="regularize")
            for name, value in (params_regularization or dict()).items()
        }
        self.params_regularization_factor = {
            name: process_function_description(value, entry_function="factor")
            for name, value in (params_regularization_factor or dict()).items()
        }
        self.scale_regularizations = scale_regularizations

    def __call__(
        self, model: MADE, inputs, trainer=None, return_factors=False
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        if not self.params_regularization:
            # shortcut for trainings with no parameter regularizations
            nll = -model.log_prob(inputs).mean()
            return (
                dict(loss=nll, nll=nll)
                if not return_factors
                else (dict(loss=nll, nll=nll), dict())
            )

        results = dict(nll=0.0, loss=0.0)
        regularizations = {name: 0.0 for name in self.params_regularization}
        for i in range(model.num_masks or 1):
            model.reorder()
            params_logits = model(inputs)
            params = model.density_estimator.transform_distribution_parameters(
                params_logits
            )
            for name, func in self.params_regularization.items():
                regularizations[name] += func(
                    params[name]
                ).mean()  # mean will ensure that regularization results are scalars
            results["nll"] -= model.density_estimator.log_prob(
                inputs, params_logits=params_logits, params=params
            ).mean()

        # normalizing by the number of masks
        factors = dict()
        results["nll"] /= model.num_masks or 1
        results["loss"] += results["nll"]
        for name in regularizations:
            regularizations[name] /= model.num_masks or 1
            results[f"params_regularization/{name}"] = regularizations[name]

            factor = self.params_regularization_factor.get(name, 1.0)
            factors[name] = (
                factor(regularizations[name], trainer=trainer, model=model)
                if callable(factor)
                else factor
            )

            results["loss"] += (
                (
                    regularizations[name]
                    / regularizations[name].data.abs()
                    * results["nll"].data
                )
                if self.scale_regularizations
                else regularizations[name]
            ) * factors[name]
        return results if not return_factors else dict(loss=nll, nll=nll), dict()
