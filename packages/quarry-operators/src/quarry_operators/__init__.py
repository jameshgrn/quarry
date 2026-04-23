"""Quarry Operators — implementations of the Operator protocol."""

from quarry_operators.aspect import AspectOperator, AspectParams
from quarry_operators.registry import OPERATOR_NAMES, get_operator, get_params_class
from quarry_operators.slope import SlopeOperator, SlopeParams

__all__ = [
    "AspectOperator",
    "AspectParams",
    "OPERATOR_NAMES",
    "SlopeOperator",
    "SlopeParams",
    "get_operator",
    "get_params_class",
]
