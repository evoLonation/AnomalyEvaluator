from typing import Callable, TypeVar, ParamSpec
import torch.nn as nn


def _returns_nn_module_call(*args):
    return nn.Module.__call__


_P = ParamSpec("_P")
_R = TypeVar("_R")


def generate_call_signature(
    forward_func: Callable[_P, _R],
) -> Callable[..., Callable[_P, _R]]:
    return _returns_nn_module_call
