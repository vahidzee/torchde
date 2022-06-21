import importlib
import typing as th
import types
import inspect

# for eval context
import torch



def safe_function_call_wrapper(function: th.Callable):
    signature = inspect.signature(function)
    params = signature.parameters

    def wrapper(*args, **kwargs):
        call_kwargs = {name: kwargs[name] for name in params if name in kwargs}
        return function(*args, **call_kwargs)

    return wrapper


def generate_function(code_block, function: str) -> th.Callable[[th.Any], th.Any]:
    context = dict()
    exec(code_block, dict(), context)
    return types.FunctionType(
        code=context[function].__code__,
        globals=context,
        name=function,
        argdefs=context[function].__defaults__,
    )


def process_function_description(
    function: th.Union[th.Callable, str, th.Dict[str, str]], entry_function
) -> th.Callable:
    if callable(function) or not isinstance(function, (str, dict)):
        return function
    try:
        return eval(function if isinstance(function, str) else function["code"])
    except SyntaxError:
        return generate_function(
            code_block=function if isinstance(function, str) else function["code"],
            function=function.get("entry", entry_function) if isinstance(function, dict) else entry_function,
        )


def import_context(name: str):
    return importlib.import_module(name)


def get_value(name: str, context: th.Optional[th.Any] = None, strict: bool = True):
    var = context if context is not None else import_context(name.split(".")[0])
    for split in name.split(".")[(0 if context is not None else 1) :]:
        if isinstance(var, dict):
            if split not in var:
                if strict:
                    raise KeyError('Invalid key "%s"' % name)
                else:
                    return None
            var = var[split]
        else:
            if not hasattr(var, split):
                if strict:
                    raise AttributeError("Invalid attribute %s" % name)
                else:
                    return None
            var = getattr(var, split)
    return var
