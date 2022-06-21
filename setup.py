from setuptools import setup, find_packages
from importlib.util import module_from_spec, spec_from_file_location
import typing as th
import re
import types
import os


# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_MADE_SRC = os.path.join(_PATH_ROOT, "mdade")
_PATH_REQUIREMENTS = os.path.join(_PATH_ROOT, "requirements")

# read description
with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# adapted from https://github.com/Lightning-AI/lightning/blob/master/setup.py
def _load_py_module(name: str, location: str) -> types.ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _load_requirements(path_dir: str, file_name: str, comment_char: str = "#", unfreeze: bool = True) -> th.List[str]:
    """Load requirements from a file.
    >>> _load_requirements(_PATH_REQUIREMENTS)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        comment = ""
        if comment_char in ln:
            comment = ln[ln.index(comment_char) :]
            ln = ln[: ln.index(comment_char)]
        req = ln.strip()
        # skip directly installed dependencies
        if not req or req.startswith("http") or "@http" in req:
            continue
        # remove version restrictions unless they are strict
        if unfreeze and "<" in req and "strict" not in comment:
            req = re.sub(r",? *<=? *[\d\.\*]+", "", req).strip()
        reqs.append(req)
    return reqs


_ABOUT_MODULE = _load_py_module(name="about", location=os.path.join(_PATH_MADE_SRC, "__about__.py"))

# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install pytorch-lightning[dev, docs]`
# From local copy of repo, use like `pip install ".[dev, docs]"`
extras = {
    "tools": _load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="tools.txt"),
}

setup(
    name="mdade",
    packages=find_packages(include=["*"]),
    version=_ABOUT_MODULE.__version__,
    description=_ABOUT_MODULE.__docs__,
    author=_ABOUT_MODULE.__author__,
    author_email=_ABOUT_MODULE.__author_email__,
    url=_ABOUT_MODULE.__homepage__,
    download_url="https://github.com/vahidzee/mdade",
    license=_ABOUT_MODULE.__license__,
    long_description=long_description,  # same as readme
    long_description_content_type="text/markdown",
    keywords=[
        "deep learning",
        "pytorch",
        "AI",
        "density estimation",
        "masked autoencoders",
        "pixelcnn",
        "autoregressive models",
    ],
    install_requires=_load_requirements(_PATH_REQUIREMENTS, "base.txt"),
    extras_require=extras,
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        # the project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # license
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": ["mdade=mdade.main:main"],
    },
)
