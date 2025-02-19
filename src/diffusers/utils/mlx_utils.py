import importlib.util
import operator as op
import os
import sys
from collections import OrderedDict
from itertools import chain
from types import ModuleType
from typing import Any, List, Optional, Tuple, Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging import version
from packaging.version import Version, parse

from . import logging

from .import_utils import (
    DIFFUSERS_SLOW_IMPORT,
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    _LazyModule,
    is_flax_available,
    is_torch_available,
    is_torch_xla_available,
)

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
    
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

USE_MLX = os.environ.get("USE_MLX", "AUTO").upper() 

if USE_MLX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _mlx_available = importlib.util.find_spec("mlx") is not None
    if _mlx_available:
        try:
            _mlx_version = importlib_metadata.version("mlx")
            logger.info(f"MLX version {_mlx_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _mlx_available = False
else:
    _mlx_available = False
    
def is_mlx_available():
    return _mlx_available


# docstyle-ignore
MLX_IMPORT_ERROR = """
{0} requires the mlx library but it was not found in your environment. Checkout the instructions on the
installation page: https://ml-explore.github.io/mlx/build/html/install.html and follow the ones that match your environment.
"""
