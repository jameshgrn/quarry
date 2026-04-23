"""Operator registry — lazy name-to-class mapping.

Lane: registry

Maps operator names to their implementation classes using importlib.
OPERATOR_NAMES is a cheap tuple (no heavy imports at module load).
get_operator() and get_params_class() trigger actual module imports on demand.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quarry_core.operator import Operator, OperatorParams

# (name, module_path, operator_class_name)
_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("aspect", "quarry_operators.aspect", "AspectOperator"),
    ("build_cog", "quarry_operators.build_cog", "BuildCOGOperator"),
    ("clip_raster", "quarry_operators.clip_raster", "ClipRasterOperator"),
    ("d8_flow_direction", "quarry_operators.d8_flow_direction", "D8FlowDirectionOperator"),
    ("fill_depressions", "quarry_operators.fill_depressions", "FillDepressionsOperator"),
    ("flow_accumulation", "quarry_operators.flow_accumulation", "FlowAccumulationOperator"),
    ("rasterize_vector", "quarry_operators.rasterize_vector", "RasterizeVectorOperator"),
    ("reproject", "quarry_operators.reproject", "ReprojectOperator"),
    ("sample_raster", "quarry_operators.sample_raster", "SampleRasterOperator"),
    ("slope", "quarry_operators.slope", "SlopeOperator"),
    ("spatial_join", "quarry_operators.spatial_join", "SpatialJoinOperator"),
    ("zonal_stats", "quarry_operators.zonal_stats", "ZonalStatsOperator"),
)

OPERATOR_NAMES: tuple[str, ...] = tuple(name for name, _, _ in _REGISTRY)

_NAME_TO_ENTRY: dict[str, tuple[str, str]] = {
    name: (module_path, class_name) for name, module_path, class_name in _REGISTRY
}


def get_operator(name: str) -> Operator:
    """Import and instantiate the operator for the given name.

    Raises KeyError if name is unknown.
    """
    if name not in _NAME_TO_ENTRY:
        raise KeyError(f"Unknown operator: {name!r}. Available: {', '.join(OPERATOR_NAMES)}")
    module_path, class_name = _NAME_TO_ENTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def get_params_class(name: str) -> type[OperatorParams]:
    """Import and return the params dataclass for the given operator name.

    Convention: params class is named <ClassName without 'Operator'>Params.
    E.g. SlopeOperator → SlopeParams, BuildCOGOperator → BuildCOGParams.

    Raises KeyError if name is unknown.
    """
    if name not in _NAME_TO_ENTRY:
        raise KeyError(f"Unknown operator: {name!r}. Available: {', '.join(OPERATOR_NAMES)}")
    module_path, class_name = _NAME_TO_ENTRY[name]
    module = importlib.import_module(module_path)
    params_class_name = class_name.replace("Operator", "Params")
    return getattr(module, params_class_name)
