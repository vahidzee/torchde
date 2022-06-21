import time

_this_year = time.strftime("%Y")
__version__ = "0.0.3"
__author__ = "Vahid Zehtab"
__author_email__ = "vdzehtab@gmail.com"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2021-{_this_year}, {__author__}."
__homepage__ = "https://github.com/vahidzee/mdade"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "'mdade' offers a lightweight PyTorch toolbox for Masked Autoregressive models"

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__homepage__",
    "__license__",
    "__version__",
    "__docs__",
]
