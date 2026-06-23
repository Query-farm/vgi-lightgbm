"""Feature handling, including LightGBM's first-class categorical support.

LightGBM can split directly on categorical features (no one-hot blow-up), which
is one of its signature strengths. To expose that over SQL we let callers pass
raw **string** feature columns: at fit, string columns are detected, integer-
encoded with a stable per-column category ordering, and their indices are passed
to LightGBM as ``categorical_feature`` so the booster learns native categorical
splits. The category orderings are stored in the model metadata so ``predict``
re-encodes new rows identically (unseen categories map to ``-1``, which LightGBM
treats as an unknown/missing category). Numeric/boolean columns pass through as
floats, with NULLs preserved as NaN (LightGBM handles missing values natively).

The ``categorical`` mask (one bool per feature, in feature order) and the
``categories`` (list of category values per categorical feature) are recorded in
the model metadata so prediction rebuilds ``X`` with the same encoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa


def is_categorical(arrow_type: pa.DataType) -> bool:
    """Whether an Arrow column type is a (string) categorical feature."""
    return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)


def categorical_mask(field_types: list[pa.DataType]) -> list[bool]:
    """One ``is_categorical`` flag per column, in order."""
    return [is_categorical(t) for t in field_types]


def detect_categoricals(table: pa.Table, feature_names: list[str]) -> list[bool]:
    """Per-feature categorical mask from the table's column types."""
    return categorical_mask([table.schema.field(n).type for n in feature_names])


def fit_categories(table: pa.Table, feature_names: list[str], cat_mask: list[bool]) -> dict[str, list[str]]:
    """Learn the sorted distinct category values for each categorical feature.

    A stable (sorted) ordering makes the integer encoding reproducible across
    fit and predict regardless of row order.
    """
    categories: dict[str, list[str]] = {}
    for name, is_cat in zip(feature_names, cat_mask, strict=True):
        if not is_cat:
            continue
        values = {str(v) for v in table.column(name).to_pylist() if v is not None}
        categories[name] = sorted(values)
    return categories


def encode_matrix(
    table: pa.Table,
    feature_names: list[str],
    cat_mask: list[bool],
    categories: dict[str, list[str]],
) -> np.ndarray:
    """Build a numeric ``X`` matrix, integer-encoding categorical columns.

    Categorical cells become the index of their value in the stored category
    ordering (unseen / NULL -> ``-1``); numeric/boolean cells become floats with
    NULL -> NaN. The result is always a float64 matrix LightGBM can consume with
    ``categorical_feature`` pointing at the categorical column indices.
    """
    n_rows = table.num_rows
    n_cols = len(feature_names)
    out = np.empty((n_rows, n_cols), dtype=float)
    for j, (name, is_cat) in enumerate(zip(feature_names, cat_mask, strict=True)):
        col = table.column(name).to_pylist()
        if is_cat:
            index = {v: i for i, v in enumerate(categories.get(name, []))}
            out[:, j] = [(-1.0 if v is None else float(index.get(str(v), -1))) for v in col]
        else:
            out[:, j] = [(float("nan") if v is None else float(v)) for v in col]
    return out


def categorical_indices(cat_mask: list[bool]) -> list[int]:
    """Indices (in feature order) of the categorical features."""
    return [i for i, c in enumerate(cat_mask) if c]


def validate_features(table: pa.Table, feature_names: list[str], *, what: str = "feature") -> None:
    """Raise a clear error if a feature column is missing or of an unsupported type.

    Supported feature types are numeric, boolean, and string (categorical).
    """
    present = set(table.schema.names)
    missing = [n for n in feature_names if n not in present]
    if missing:
        raise ValueError(
            f"missing required {what} column(s): {', '.join(missing)}; "
            f"input has columns: {', '.join(table.schema.names)}"
        )
    bad = []
    for n in feature_names:
        t = table.schema.field(n).type
        if not (
            pa.types.is_floating(t)
            or pa.types.is_integer(t)
            or pa.types.is_boolean(t)
            or is_categorical(t)
        ):
            bad.append(f"{n} ({t})")
    if bad:
        raise ValueError(
            f"{what} column(s) have unsupported types: " + ", ".join(bad) + ". "
            "Features must be numeric, boolean, or string (categorical)."
        )


def feature_types_of(table: pa.Table, feature_names: list[str]) -> dict[str, Any]:
    """A {name: arrow-type} map for the requested features (for diagnostics)."""
    return {n: table.schema.field(n).type for n in feature_names}
