import torch
import functools
import pytorch_lightning as pl
from torchde.utils import get_value, safe_function_call_wrapper, process_function_description, FunctionDescriptor
import typing as th


# types
TermDescriptor = th.Union[str, th.Dict[str, th.Any], FunctionDescriptor]
ResultsDict = "torchde.training.criterion.ResultsDict"


class CriterionTerm:
    def __init__(
        self,
        name: str = None,
        factor: th.Optional[th.Union[float, FunctionDescriptor]] = None,
        scale_factor: th.Optional[str] = None,
        term_function: th.Optional[FunctionDescriptor] = None,
        factor_application: str = "multiply",  # multiply or add
        **kwargs,  # function description dictionary
    ) -> None:
        self.name = name or self.__class__.__name__
        self._factor_application = factor_application
        self._factor_description = factor
        self.initialize_factor_attributes(factor_application, factor)
        self._term_function_description = term_function or kwargs
        self._scale_factor = scale_factor

    def initialize_factor_attributes(
        self,
        factor_application: th.Optional[str] = None,
        factor_description: th.Optional[th.Union[float, FunctionDescriptor]] = None,
    ):
        self._factor_application = factor_application or self._factor_application
        self._factor_description = factor_description or self._factor_description
        if self._factor_application == "multiply" and self._factor_description is None:
            self._factor_description = 1.0
        if self._factor_application == "add" and self._factor_description is None:
            self._factor_description = 0.0

    @functools.cached_property
    def _compiled_factor(self):
        compiled = process_function_description(self._factor_description, entry_function="factor")
        return safe_function_call_wrapper(compiled) if callable(compiled) else compiled

    def factor_value(self, results_dict: ResultsDict, training_module: pl.LightningModule) -> torch.Tensor:
        factor_value = (
            self._compiled_factor(results_dict=results_dict, training_module=training_module)
            if callable(self._compiled_factor)
            else self._compiled_factor
        )
        return (
            factor_value
            if not self._scale_factor
            else factor_value * results_dict[self._scale_factor].data.clone() / results_dict[self.name].data.clone()
        )

    def apply_factor(
        self,
        term_value: th.Optional[torch.Tensor] = None,
        factor_value: th.Optional[torch.Tensor] = None,
        term_results: th.Optional[ResultsDict] = None,
        training_module: th.Optional[pl.LightningModule] = None,
    ) -> torch.Tensor:
        factor_value = (
            self.factor_value(term_results=term_results, training_module=training_module)
            if factor_value is None
            else factor_value
        )
        term_value = term_results[self.name] if term_value is None else term_value
        if self._factor_application == "multiply":
            return term_value * factor_value
        elif self._factor_application == "add":
            return term_value + factor_value
        else:
            raise ValueError(f"Unknown factor application {self._factor_application}")

    @functools.cached_property
    def _compiled_term_function(self):
        compiled = process_function_description(self._term_function_description, entry_function="term")
        return safe_function_call_wrapper(compiled) if callable(compiled) else compiled

    def __call__(
        self, *args, inputs: th.Any = None, training_module: pl.LightningModule = None, **kwargs
    ) -> torch.Tensor:
        if not self._term_function_description:
            raise NotImplementedError
        return self._compiled_term_function(*args, inputs=inputs, training_module=training_module, **kwargs).mean()

    @staticmethod
    def from_description(
        description: th.Union["CriterionTerm", TermDescriptor],
        # overwrite attributes of the instance
        name: th.Optional[str] = None,
        factor_application: th.Optional[str] = None,  # multiply or addÍ
    ) -> "CriterionTerm":
        if isinstance(description, CriterionTerm):
            term = description
        elif isinstance(description, str):
            try:
                term = get_value(description)()
            except:
                term = CriterionTerm(term_function=description, name=name, factor_application=factor_application)
        # else the description is a dict
        # checking if the description provides a class_path to instantiate a previously defined term
        elif "class_path" in description:
            term = get_value(description["class_path"])(**description.get("init_args", dict()))
        # else the description is a dict with required fields to instantiate a new term
        else:
            term = CriterionTerm(**description)
        if name is not None:
            term.name = name
        if factor_application is not None:
            term.initialize_factor_attributes(factor_application=factor_application)
        return term
