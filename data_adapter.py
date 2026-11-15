"""Adapter for an authorized, externally stored dataset preprocessing module.

The industrial data and its site-specific paths are intentionally kept outside
the public code release.  Set BCAE_DATASET_MODULE_DIR to a directory containing
preprocess.py, which must expose ld_dataset and tep_dataset.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Optional, Union


DATASET_MODULE_ENV = "BCAE_DATASET_MODULE_DIR"


def _load_preprocess_module(
    data_module_dir: Optional[Union[str, os.PathLike]] = None,
) -> ModuleType:
    module_dir = Path(
        data_module_dir or os.environ.get(DATASET_MODULE_ENV, "")
    ).expanduser()

    if not str(module_dir) or str(module_dir) == ".":
        raise RuntimeError(
            "Dataset module directory is not configured. Set "
            f"{DATASET_MODULE_ENV} to the directory containing preprocess.py."
        )

    module_path = module_dir.resolve() / "preprocess.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Expected an authorized preprocessing module at {module_path}."
        )

    spec = importlib.util.spec_from_file_location(
        "bcae_external_preprocess", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load preprocessing module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_loader(name, *args, data_module_dir=None, **kwargs):
    module = _load_preprocess_module(data_module_dir)
    loader = getattr(module, name, None)
    if loader is None:
        raise AttributeError(
            f"{module.__file__} does not define the required function {name}."
        )
    return loader(*args, **kwargs)


def ld_dataset(*args, data_module_dir=None, **kwargs):
    """Load the pipeline dataset through the authorized external module."""
    return _call_loader(
        "ld_dataset", *args, data_module_dir=data_module_dir, **kwargs
    )


def tep_dataset(*args, data_module_dir=None, **kwargs):
    """Load the TEP dataset through the authorized external module."""
    return _call_loader(
        "tep_dataset", *args, data_module_dir=data_module_dir, **kwargs
    )
