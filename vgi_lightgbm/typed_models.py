"""Typed per-estimator fit functions: ``lightgbm.fit_lgbm_<task>(...)``.

These wrap the generic ``fit`` with LightGBM's common hyperparameters exposed as
**native, typed SQL named arguments** — so they show up in the catalog and
DuckDB's autocomplete, are type-checked, and are discoverable without consulting
docs:

    SELECT * FROM lightgbm.fit_lgbm_classifier(
      (SELECT * FROM training), model_name := 'm', target := 'y',
      n_estimators := 300, num_leaves := 63, learning_rate := 0.05);

Each function behaves exactly like ``fit``: it returns the training summary plus
the model as a BLOB, and persists to the registry when ``model_name`` is given.
The generic ``fit`` (JSON ``params``) remains the escape hatch for hyperparameters
not surfaced here. The curated parameter set is the common, high-value LightGBM
knobs — see ``_HPARAMS`` below.

Sentinels: ``max_depth := 0`` maps to ``-1`` (LightGBM's "unlimited"), and an
empty ``objective := ''`` keeps LightGBM's task default.
"""

from __future__ import annotations

import types
from dataclasses import field as dc_field
from dataclasses import make_dataclass
from typing import Annotated, Any

from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of
from .models import _ESTIMATORS, _FIT_SCHEMA, _fit_and_emit
from .registry import validate_name
from .schema_utils import columns_md

_UNSET: Any = object()

# All typed fit_lgbm_<task> functions share the generic fit result schema.
_FIT_COLUMNS_MD = columns_md(_FIT_SCHEMA)


class _HP:
    """One typed hyperparameter exposed as a SQL named argument."""

    __slots__ = ("name", "type", "default", "doc", "none_if", "map_value")

    def __init__(
        self,
        name: str,
        type: type,
        default: Any,
        doc: str,
        *,
        none_if: Any = _UNSET,
        map_value: Any = None,
    ) -> None:
        self.name = name
        self.type = type
        self.default = default
        self.doc = doc
        self.none_if = none_if  # if the SQL value equals this, pass None to LightGBM
        self.map_value = map_value  # optional callable: SQL value -> LightGBM value


# Common, high-value LightGBM hyperparameters shared by both tasks. The only
# difference between classifier and regressor is the default ``objective``.
def _common(default_objective: str) -> list[_HP]:
    return [
        _HP("n_estimators", int, 100, "Number of boosting iterations (trees)."),
        _HP("num_leaves", int, 31, "Max leaves per tree (the main capacity knob)."),
        _HP("max_depth", int, 0, "Max tree depth; 0 = unlimited.", none_if=0, map_value=lambda v: -1),
        _HP("learning_rate", float, 0.1, "Shrinkage applied to each tree."),
        _HP("min_child_samples", int, 20, "Min data in a leaf (regularization)."),
        _HP("subsample", float, 1.0, "Row subsampling fraction (bagging_fraction)."),
        _HP("colsample_bytree", float, 1.0, "Column subsampling fraction per tree."),
        _HP("reg_alpha", float, 0.0, "L1 regularization."),
        _HP("reg_lambda", float, 0.0, "L2 regularization."),
        _HP("boosting_type", str, "gbdt", "Boosting algorithm: gbdt, dart, or goss."),
        _HP("objective", str, default_objective, "LightGBM objective; '' keeps the task default.", none_if=""),
        _HP("random_state", int, 0, "Random seed."),
    ]


_HPARAMS: dict[str, list[_HP]] = {
    "lgbm_classifier": _common(""),
    "lgbm_regressor": _common("regression"),
}

# Per-estimator prose for the typed fit_lgbm_<task> functions: a one-line task
# summary woven into the rich doc tags below so each generated function gets
# distinct, estimator-specific documentation rather than a templated string.
_ESTIMATOR_DOC: dict[str, tuple[str, str]] = {
    "lgbm_classifier": (
        "gradient-boosted decision-tree **classifier** (LightGBM's `LGBMClassifier`)",
        "classification",
    ),
    "lgbm_regressor": (
        "gradient-boosted decision-tree **regressor** (LightGBM's `LGBMRegressor`)",
        "regression",
    ),
}


def _typed_doc_tags(est_name: str) -> dict[str, str]:
    """Build distinct, estimator-specific ``vgi.doc_llm`` / ``vgi.doc_md`` tags."""
    blurb, task = _ESTIMATOR_DOC[est_name]
    doc_llm = (
        f"Buffers a training table and fits a {blurb} with its key hyperparameters exposed as typed, "
        f"named SQL arguments (`n_estimators`, `num_leaves`, `max_depth`, `learning_rate`, "
        f"`min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, "
        f"`boosting_type`, ...). Equivalent to `fit(estimator := '{est_name}', ...)` but discoverable and "
        f"type-checked in the catalog instead of a JSON `params` blob. Name the {task} label with "
        f"`target :=`, optionally carry an `id :=` through; every other column is a feature (string "
        f"columns become LightGBM native categoricals and missing values flow through). Returns the same "
        f"one-row summary with a reusable `model` BLOB, and persists to the registry when `model_name :=` "
        f"is given. Feed the BLOB to `predict`/`explain`/`feature_importance`. Sentinel: `max_depth := 0` "
        f"means unlimited."
    )
    doc_md = (
        f"**`fit_{est_name}`** — fit a {blurb} with typed hyperparameters.\n\n"
        f"- Input: a training table `(SELECT ...)`; name the label with `target :=`, optional `id :=` "
        f"passthrough\n"
        f"- Hyperparameters are named, typed SQL args (`n_estimators`, `num_leaves`, `max_depth`, "
        f"`learning_rate`, `min_child_samples`, `reg_lambda`, ...) at LightGBM's defaults\n"
        f"- Returns the standard fit summary plus a reusable `model` BLOB; `model_name :=` also persists "
        f"it\n\n"
        f"The discoverable, type-checked alternative to `fit(estimator := '{est_name}', ...)`. "
        f"`max_depth := 0` = unlimited."
    )
    return {"vgi.result_columns_md": _FIT_COLUMNS_MD, "vgi.doc_llm": doc_llm, "vgi.doc_md": doc_md}


def _estimator_kwargs(spec: list[_HP], args: Any) -> dict[str, Any]:
    """Translate the typed SQL args into LightGBM estimator kwargs."""
    kw: dict[str, Any] = {}
    for hp in spec:
        v = getattr(args, hp.name)
        if hp.none_if is not _UNSET and v == hp.none_if:
            v = None if hp.map_value is None else hp.map_value(v)
        kw[hp.name] = v
    return kw


def _make_args_class(est_name: str, spec: list[_HP]) -> type:
    fields: list[Any] = [
        ("data", Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]),
        (
            "model_name",
            Annotated[
                str,
                Arg("model_name", default="", doc="Optional registry name; the model is always returned as a BLOB."),
            ],
            dc_field(default=""),
        ),
        (
            "target",
            Annotated[str, Arg("target", default="", doc="Label column name (required).")],
            dc_field(default=""),
        ),
        ("id", Annotated[str, Arg("id", default="", doc="Optional id passthrough column.")], dc_field(default="")),
    ]
    for hp in spec:
        fields.append(
            (hp.name, Annotated[hp.type, Arg(hp.name, default=hp.default, doc=hp.doc)], dc_field(default=hp.default))
        )
    cls_name = "Fit" + "".join(p.title() for p in est_name.split("_")) + "Args"
    return make_dataclass(cls_name, fields, frozen=True, slots=True)


def _make_fit_function(est_name: str) -> type:
    task, est_cls, defaults = _ESTIMATORS[est_name]
    spec = _HPARAMS[est_name]
    args_cls = _make_args_class(est_name, spec)
    fn_name = f"fit_{est_name}"
    param_hint = ", ".join(f"{hp.name} := {hp.default!r}" for hp in spec[:2])

    def on_bind(cls: type, params: BindParams[Any]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError(f"{fn_name} requires 'target' (the label column name, e.g. target := 'label')")
        if a.model_name:
            validate_name(a.model_name)
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_FIT_SCHEMA)

    def initial_finalize_state(cls: type, finalize_state_id: bytes, params: TableBufferingParams[Any]) -> DrainState:
        return DrainState()

    def finalize(
        cls: type,
        params: TableBufferingParams[Any],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        kwargs = _estimator_kwargs(spec, a)
        estimator = est_cls(**{**defaults, **kwargs})
        input_schema = input_schema_of(params)
        table = cls.buffered_table(params, input_schema)  # type: ignore[attr-defined]
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema,
            estimator_label=est_name,
            task=task,
            estimator=estimator,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict={k: v for k, v in kwargs.items() if v is not None},
        )

    meta = type(
        "Meta",
        (),
        {
            "name": fn_name,
            "description": f"Fit a {est_name} with typed LightGBM hyperparameters; returns/stores the model",
            "categories": ["models", "supervised", "typed"],
            "tags": _typed_doc_tags(est_name),
            "examples": [
                FunctionExample(
                    sql=(
                        f"SELECT model_name, task FROM lightgbm.{fn_name}((SELECT * FROM training), "
                        f"model_name := 'm', target := 'y'" + (f", {param_hint}" if param_hint else "") + ")"
                    ),
                    description=f"Train a {est_name} with named hyperparameters",
                )
            ],
        },
    )
    namespace = {
        "FunctionArguments": args_cls,
        "Meta": meta,
        "on_bind": classmethod(on_bind),
        "initial_finalize_state": classmethod(initial_finalize_state),
        "finalize": classmethod(finalize),
    }
    cls_name = "Fit" + "".join(p.title() for p in est_name.split("_"))
    return types.new_class(cls_name, (SinkBuffer[args_cls, DrainState],), {}, lambda ns: ns.update(namespace))  # type: ignore[valid-type]


TYPED_FIT_FUNCTIONS: list[type] = [_make_fit_function(name) for name in _HPARAMS]
