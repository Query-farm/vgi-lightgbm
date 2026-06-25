"""Supervised learning: fit LightGBM estimators into the registry and predict from it.

* ``fit``       -- TableBufferingFunction: buffer the training table, fit an
  estimator, **return the model as a BLOB**, and persist it to the registry when
  ``model_name`` is given (so ``model_name`` is optional).
* ``predict``   -- TableInOutGenerator: stream a table through a model named in the
  registry (``model_name :=``) *or* passed inline as a BLOB (``model :=``).
* ``cross_val_predict`` -- buffering: out-of-fold predictions, no persistence.
* ``cross_val_score``   -- buffering: per-fold scores, no persistence.
* ``list_models`` / ``model_info`` / ``drop_model`` -- registry management.

Column roles follow the project convention: name the ``target`` column (for
fit / cross-val) and optionally an ``id`` column to carry through; every other
column is a feature. Features may be numeric, boolean, or **string** — string
columns are passed to LightGBM as native categorical features (its signature
strength). NULLs are preserved as missing values, which LightGBM handles
natively. Hyperparameters are passed as a JSON string (the typed
``fit_lgbm_*`` functions expose the common ones as named args).

    SELECT * FROM lightgbm.fit((SELECT * FROM training), model_name := 'iris_clf',
                               estimator := 'lgbm_classifier', target := 'species', id := 'id');
    SELECT * FROM lightgbm.predict((SELECT * FROM new_data), model_name := 'iris_clf', id := 'id');
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import lightgbm as lgb
import numpy as np
import pyarrow as pa
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import OutputCollector as InOutCollector
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.log import Level

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import (
    categorical_indices,
    detect_categoricals,
    encode_matrix,
    fit_categories,
    validate_features,
)
from .registry import (
    ModelBlobError,
    ModelMetadata,
    ModelNotFoundError,
    booster_from_text,
    get_store,
    now_iso,
    pack_model,
    unpack_meta,
    unpack_model,
    validate_name,
)
from .schema_utils import columns_md, columns_md_rows
from .schema_utils import field as sfield

CLASSIFICATION = "classification"
REGRESSION = "regression"

# A "quiet" default so LightGBM does not spew training logs over the RPC stream.
_QUIET = {"random_state": 0, "verbosity": -1, "n_jobs": 1}

# name -> (task, estimator class, default kwargs)
_ESTIMATORS: dict[str, tuple[str, type, dict[str, Any]]] = {
    "lgbm_classifier": (CLASSIFICATION, LGBMClassifier, dict(_QUIET)),
    "lgbm_regressor": (REGRESSION, LGBMRegressor, dict(_QUIET)),
}


def _parse_params(params: str) -> dict[str, Any]:
    params = (params or "").strip()
    if not params:
        return {}
    parsed = json.loads(params)
    if not isinstance(parsed, dict):
        raise ValueError("params must be a JSON object, e.g. '{\"n_estimators\": 200}'")
    return parsed


def estimator_param_names(name: str) -> list[str]:
    """Sorted list of hyperparameters accepted by an estimator (for discovery/errors)."""
    _task, cls, _defaults = _ESTIMATORS[name]
    return sorted(cls().get_params().keys())


def build_estimator(name: str, params: dict[str, Any]) -> tuple[str, Any]:
    """Return ``(task, estimator)`` for a registered estimator name + hyperparams."""
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
    task, cls, defaults = _ESTIMATORS[name]
    # Reject unknown hyperparameters up front with the valid set, rather than
    # surfacing LightGBM's opaque error later.
    valid = set(cls().get_params().keys())
    unknown = [k for k in params if k not in valid]
    if unknown:
        raise ValueError(
            f"unknown hyperparameter(s) for {name!r}: {', '.join(sorted(unknown))}. "
            f"valid params: {', '.join(sorted(valid))}"
        )
    try:
        return task, cls(**{**defaults, **params})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid hyperparameters for {name!r}: {exc}") from exc


def _features_excluding(input_schema: pa.Schema, *exclude: str) -> list[str]:
    drop = {e for e in exclude if e}
    return [n for n in input_schema.names if n not in drop]


def _label_dtype(classes: list[Any] | None) -> pa.DataType:
    """Arrow type for a decoded prediction column, from the original class labels."""
    if classes and any(isinstance(c, str) for c in classes):
        return pa.string()
    return pa.int64()


def _prediction_field(task: str, classes: list[Any] | None = None) -> pa.Field:
    if task == CLASSIFICATION:
        return sfield("prediction", _label_dtype(classes), "Predicted class label.", nullable=False)
    return sfield("prediction", pa.float64(), "Predicted value.", nullable=False)


def _proba_label(c: Any) -> str:
    """Column-name-safe rendering of a class label for proba_<label> columns."""
    return str(c)


def encode_labels(values: list[Any]) -> tuple[np.ndarray, list[Any]]:
    """Label-encode raw target values to 0..n-1 integer codes.

    Returns ``(codes, classes)`` where ``classes`` is the ordered list of the
    *original* labels (so ``classes[code]`` decodes a prediction). Labels are
    ordered as scikit-learn's ``LabelEncoder`` does: sorted distinct values. Any
    hashable/orderable label type works (int, float, string, bool).
    """
    seen = [v for v in values if v is not None]
    if not seen:
        raise ValueError("classification target has no non-null labels")
    try:
        classes = sorted(set(seen), key=lambda v: (str(type(v)), v))
    except TypeError as exc:  # pragma: no cover - mixed unorderable labels
        raise ValueError(f"classification target labels are not orderable: {exc}") from exc
    index = {c: i for i, c in enumerate(classes)}
    codes = np.array([index[v] for v in values], dtype=int)
    return codes, classes


def _target_array(table: pa.Table, target: str, task: str) -> np.ndarray:
    """Numeric target for regression; for classification use ``_target_codes``."""
    col = table.column(target)
    if task == CLASSIFICATION:
        codes, _classes = _target_codes(table, target)
        return codes
    try:
        return np.asarray(col.to_numpy(zero_copy_only=False)).astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"regression target {target!r} must be numeric; could not convert its values to numbers ({exc})"
        ) from exc


def _target_codes(table: pa.Table, target: str) -> tuple[np.ndarray, list[Any]]:
    """Label-encode a classification target column to codes + ordered original labels."""
    values = table.column(target).to_pylist()
    return encode_labels(values)


# ===========================================================================
# Shared fit core (used by both `fit` and the typed `fit_lgbm_*` functions)
# ===========================================================================

_FIT_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name the model was stored under (NULL if not persisted)."),
        sfield("estimator", pa.string(), "Estimator type used.", nullable=False),
        sfield("task", pa.string(), "classification or regression.", nullable=False),
        sfield("n_samples", pa.int64(), "Number of training rows.", nullable=False),
        sfield("n_features", pa.int64(), "Number of features.", nullable=False),
        sfield("n_classes", pa.int64(), "Number of classes (NULL for regression)."),
        sfield("n_categorical", pa.int64(), "Number of categorical (string) features.", nullable=False),
        sfield("train_score", pa.float64(), "In-sample score (accuracy or R^2)."),
        sfield("features", pa.list_(pa.string()), "Ordered feature column names.", nullable=False),
        sfield("model", pa.large_binary(), "The fitted model as a portable BLOB.", nullable=False),
    ]
)


def _fit_and_emit(
    out: OutputCollector,
    output_schema: pa.Schema,
    *,
    table: pa.Table | None,
    input_schema: pa.Schema,
    estimator_label: str,
    task: str,
    estimator: Any,
    model_name: str,
    target: str,
    id_col: str,
    params_dict: dict[str, Any],
) -> None:
    """Fit ``estimator`` on the buffered table, persist if named, and emit the summary + BLOB."""
    if table is None or table.num_rows == 0:
        raise ValueError("fit received no training rows")

    feats = _features_excluding(input_schema, target, id_col)
    validate_features(table, feats)
    cat_mask = detect_categoricals(table, feats)
    categories = fit_categories(table, feats, cat_mask)
    cat_idx = categorical_indices(cat_mask)

    x = encode_matrix(table, feats, cat_mask, categories)
    if task == CLASSIFICATION:
        y, classes = _target_codes(table, target)
    else:
        y = _target_array(table, target, task)
        classes = None

    fit_kwargs: dict[str, Any] = {}
    if cat_idx:
        fit_kwargs["categorical_feature"] = cat_idx
    estimator.fit(x, y, **fit_kwargs)
    train_score = float(estimator.score(x, y))
    # estimator.classes_ are the 0..n-1 codes; the original labels live in `classes`.

    meta = ModelMetadata(
        name=model_name or "",
        estimator=estimator_label,
        task=task,
        target=target,
        feature_names=feats,
        params=params_dict,
        classes=classes,
        categorical=cat_mask,
        categories=categories,
        n_samples=int(table.num_rows),
        n_features=len(feats),
        train_score=train_score,
        lightgbm_version=lgb.__version__,
        created_at=now_iso(),
    )
    blob = pack_model(estimator, meta)
    if model_name:
        get_store().save(estimator, meta)

    out.emit(
        pa.RecordBatch.from_pydict(
            {
                "model_name": [model_name or None],
                "estimator": [estimator_label],
                "task": [task],
                "n_samples": [meta.n_samples],
                "n_features": [meta.n_features],
                "n_classes": [len(classes) if classes is not None else None],
                "n_categorical": [len(cat_idx)],
                "train_score": [train_score],
                "features": [feats],
                "model": [blob],
            },
            schema=output_schema,
        )
    )


# ===========================================================================
# fit
# ===========================================================================


# Optional string args carry a "" default so an omitted value reaches on_bind as
# "" and we can raise a friendly error or treat it as unset.
@dataclass(slots=True, frozen=True)
class FitArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Optional registry name; the model is always returned as a BLOB.")
    ]
    estimator: Annotated[str, Arg("estimator", default="lgbm_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]


class FitModel(SinkBuffer[FitArgs, DrainState]):
    FunctionArguments: ClassVar[type] = FitArgs

    class Meta:
        name = "fit"
        description = "Fit a LightGBM estimator; returns the model as a BLOB and stores it when model_name is given"
        categories = ["models", "supervised"]
        tags = {
            "vgi.result_columns_md": columns_md(_FIT_SCHEMA),
            "vgi.doc_llm": (
                "Buffers a training table, fits a LightGBM estimator (`estimator :=` one of "
                "`lgbm_classifier` or `lgbm_regressor`), and returns a one-row training summary whose "
                "`model` column is a self-contained BLOB (booster text + metadata). Name the label column "
                "with `target :=`; every other column except an optional `id :=` passthrough becomes a "
                "feature — string columns are used as LightGBM's native categorical features and NULLs are "
                "kept as missing values. Hyperparameters go in a JSON `params :=` string. Pass "
                "`model_name :=` to also persist the model to the registry; otherwise it lives only in the "
                "returned BLOB. Feed that BLOB to `predict`/`explain`/`feature_importance` via "
                "`SET VARIABLE` + `getvariable()`."
            ),
            "vgi.doc_md": (
                "**Fit a LightGBM model** — train and return a reusable model BLOB.\n\n"
                "- Input: a training table `(SELECT ...)`; name the label with `target :=`, optionally an "
                "`id :=` passthrough\n"
                "- `estimator :=` `lgbm_classifier` | `lgbm_regressor`; hyperparameters via JSON "
                "`params :=`\n"
                "- Returns one row: `estimator`, `task`, `n_samples`/`n_features`/`n_classes`/"
                "`n_categorical`, `train_score`, `features`, and the `model` BLOB\n"
                "- `model_name :=` also persists to the registry; string features are learned as native "
                "categoricals and classification labels of any dtype are decoded back on predict"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT model_name, task, n_samples FROM lightgbm.fit("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM lightgbm.iris()), model_name := 'iris_clf', "
                    "estimator := 'lgbm_classifier', target := 'target', id := 'sample_id')"
                ),
                description="Train a LightGBM classifier on iris and store it as 'iris_clf'",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FitArgs]) -> BindResponse:
        a = params.args
        if a.model_name:
            validate_name(a.model_name)
        if not a.target:
            raise ValueError("fit requires 'target' (the label column name, e.g. target := 'label')")
        # Validate estimator + hyperparameters now so errors surface at bind.
        build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_FIT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FitArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FitArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        task, estimator = build_estimator(a.estimator, _parse_params(a.params))
        table = cls.buffered_table(params, input_schema)
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema,
            estimator_label=a.estimator,
            task=task,
            estimator=estimator,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict=_parse_params(a.params),
        )


# ===========================================================================
# predict
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PredictArgs:
    data: Annotated[TableInput, Arg(0, doc="Table to score (must contain the model's feature columns).")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Name of a stored model (or pass model:=).")]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB from fit() (alternative to model_name).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]
    with_proba: Annotated[
        bool, Arg("with_proba", default=False, doc="Also emit per-class probabilities (classifiers).")
    ]
    output_margin: Annotated[
        bool,
        Arg("output_margin", default=False, doc="Emit the raw (untransformed) margin score instead of the label."),
    ]
    pred_leaf: Annotated[
        bool,
        Arg("pred_leaf", default=False, doc="Emit the leaf index each tree assigns the row (a list per row)."),
    ]


# Loaded models cached per query execution to avoid reloading each batch.
_PREDICT_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}
# Execution ids for which a version-mismatch warning was already emitted.
_VERSION_WARNED: set[bytes] = set()


def _meta_for_predict(a: PredictArgs) -> ModelMetadata:
    """Load just the metadata for a predict/explain request (model BLOB or name)."""
    if a.model:
        return unpack_meta(a.model)
    if not a.model_name:
        raise ValueError("predict requires either 'model_name' or a 'model' BLOB (e.g. model_name := 'my_model')")
    try:
        return get_store().load_meta(a.model_name)
    except ModelNotFoundError as exc:
        raise ValueError(f"model {a.model_name!r} not found in the registry") from exc


def _load_model(a: Any) -> tuple[lgb.Booster, ModelMetadata]:
    if getattr(a, "model", b""):
        return unpack_model(a.model)
    return get_store().load(a.model_name)


def _proba(booster: lgb.Booster, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class probabilities from a raw booster (binary boosters emit P(class=1))."""
    raw = booster.predict(x)
    raw = np.asarray(raw)
    if n_classes <= 2:
        p1 = raw.reshape(-1)
        return np.column_stack([1.0 - p1, p1])
    return raw


class PredictModel(TableInOutGenerator[PredictArgs]):
    FunctionArguments: ClassVar[type] = PredictArgs

    class Meta:
        name = "predict"
        description = "Score a table through a stored model (model_name) or an inline model BLOB"
        categories = ["models", "supervised", "inference"]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    (
                        "prediction",
                        "BIGINT, VARCHAR, or DOUBLE",
                        "Predicted class label (classification) or value (regression).",
                    ),
                ],
                note=(
                    "If an `id` column is named, it is carried through as the first column. The middle column "
                    "varies with the prediction mode: default is `prediction` (typed from the label dtype -- "
                    "BIGINT or VARCHAR for classification, DOUBLE for regression); `output_margin := true` emits a "
                    "`margin` DOUBLE; `pred_leaf := true` emits a `leaf` INTEGER[] (one leaf index per tree). "
                    "With `with_proba := true` on a classifier, one `proba_<label>` DOUBLE column is added per class."
                ),
            ),
            "vgi.doc_llm": (
                "Streams a table through an already-fit LightGBM model and emits its predictions. Identify "
                "the model with either `model_name :=` (a registry name) or `model :=` (a BLOB from `fit`, "
                "passed via `SET VARIABLE` + `getvariable()` since a table function has only one subquery "
                "slot). Features are matched by name (order-independent; extra columns ignored; missing "
                "ones error at bind), and string/categorical columns are re-encoded exactly as at fit. The "
                "default output is the decoded `prediction` (the original label dtype for classifiers, "
                "value for regressors); the mutually exclusive modes `with_proba`, `output_margin`, and "
                "`pred_leaf` switch to per-class probabilities, the raw margin, or per-tree leaf indices "
                "respectively. Name an `id :=` column to carry it through."
            ),
            "vgi.doc_md": (
                "**Predict with a stored model** — score a table row by row.\n\n"
                "- Identify the model with `model_name :=` *or* `model :=` (a `fit` BLOB)\n"
                "- Features aligned by name; an optional `id :=` is carried through\n"
                "- Default output: `prediction` (label or value)\n"
                "- `with_proba := true` → one `proba_<label>` column per class; `output_margin := true` → "
                "a `margin` DOUBLE; `pred_leaf := true` → a `leaf` INTEGER[] (one index per tree) — the "
                "three modes are mutually exclusive"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM lightgbm.predict((SELECT * FROM lightgbm.iris()), "
                    "model_name := 'iris_clf', id := 'sample_id')"
                ),
                description="Predict with the stored 'iris_clf' model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[PredictArgs]) -> BindResponse:
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _meta_for_predict(a)

        # Fail fast at bind if the input is missing any feature the model needs.
        # (predict selects features by name, so order doesn't matter and extra
        # columns are ignored — only missing ones are an error.)
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires feature column(s) {', '.join(missing)} "
                f"not present in the input; model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )

        if a.with_proba and (a.output_margin or a.pred_leaf):
            raise ValueError("with_proba cannot be combined with output_margin or pred_leaf")
        if a.output_margin and a.pred_leaf:
            raise ValueError("output_margin and pred_leaf are mutually exclusive")

        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        if a.pred_leaf:
            fields.append(
                sfield("leaf", pa.list_(pa.int32()), "Leaf index this row reaches in each tree.", nullable=False)
            )
        elif a.output_margin:
            fields.append(sfield("margin", pa.float64(), "Raw (untransformed) margin score.", nullable=False))
        else:
            fields.append(_prediction_field(meta.task, meta.classes))
        if a.with_proba and meta.task == CLASSIFICATION:
            for c in meta.classes or []:
                fields.append(sfield(f"proba_{_proba_label(c)}", pa.float64(), f"P(class = {c}).", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[PredictArgs]) -> tuple[Any, ModelMetadata]:
        assert params.init_response is not None
        key = params.init_response.execution_id
        cached = _PREDICT_CACHE.get(key)
        if cached is None:
            cached = _load_model(params.args)
            _PREDICT_CACHE[key] = cached
        return cached

    @classmethod
    def process(
        cls,
        params: ProcessParams[PredictArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        booster, meta = cls._model(params)

        assert params.init_response is not None
        key = params.init_response.execution_id
        if meta.lightgbm_version and meta.lightgbm_version != lgb.__version__ and key not in _VERSION_WARNED:
            _VERSION_WARNED.add(key)
            with contextlib.suppress(Exception):
                out.client_log(
                    Level.WARN,
                    f"model was fitted with lightgbm {meta.lightgbm_version}, "
                    f"worker has {lgb.__version__}; predictions may differ",
                )

        table = pa.Table.from_batches([batch])
        x = encode_matrix(table, meta.feature_names, meta.categorical, meta.categories)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()

        if a.pred_leaf:
            leaves = np.atleast_2d(np.asarray(booster.predict(x, pred_leaf=True)))
            columns["leaf"] = [[int(v) for v in row] for row in leaves]
        elif a.output_margin:
            # raw_score=True gives the untransformed margin; multiclass is 2D -> take the max.
            margin = np.asarray(booster.predict(x, raw_score=True))
            if margin.ndim > 1:
                margin = margin.max(axis=1)
            columns["margin"] = [float(v) for v in margin.reshape(-1)]
        elif meta.task == CLASSIFICATION:
            classes = meta.classes or []
            proba = _proba(booster, x, len(classes))
            idx = np.argmax(proba, axis=1)
            columns["prediction"] = [classes[i] for i in idx]
            if a.with_proba:
                for j, c in enumerate(classes):
                    columns[f"proba_{_proba_label(c)}"] = [float(v) for v in proba[:, j]]
        else:
            preds = np.asarray(booster.predict(x)).reshape(-1)
            columns["prediction"] = [float(v) for v in preds]

        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# cross_val_predict (no persistence)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class CrossValArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[str, Arg("estimator", default="lgbm_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]


def _cv_splitter(task: str, cv: int, y: np.ndarray) -> Any:
    if task == CLASSIFICATION:
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=0).split(np.zeros(len(y)), y)
    return KFold(n_splits=cv, shuffle=True, random_state=0).split(np.zeros(len(y)))


def _fit_fold(estimator_name: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray, cat_idx: list[int]) -> Any:
    _task, est = build_estimator(estimator_name, params)
    fit_kwargs = {"categorical_feature": cat_idx} if cat_idx else {}
    est.fit(x, y, **fit_kwargs)
    return est


class CrossValPredict(SinkBuffer[CrossValArgs, DrainState]):
    FunctionArguments: ClassVar[type] = CrossValArgs

    class Meta:
        name = "cross_val_predict"
        description = "Out-of-fold cross-validated predictions (no model is stored)"
        categories = ["models", "supervised", "evaluation"]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    (
                        "prediction",
                        "BIGINT, VARCHAR, or DOUBLE",
                        "Out-of-fold predicted class label (classification) or value (regression).",
                    ),
                ],
                note="If an `id` column is named, it is carried through as the first column.",
            ),
            "vgi.doc_llm": (
                "Computes out-of-fold cross-validated predictions for a training table without persisting "
                "any model. Each row is predicted by a model trained on the other folds (`cv :=` folds, "
                "default 5), so the result is an honest, leakage-free prediction per row — ideal for "
                "building a held-out prediction column to score with metrics or to stack. Name the label "
                "with `target :=`, pick the `estimator :=`, optionally carry an `id :=` through, and tune "
                "via JSON `params :=`. Classification labels of any dtype are decoded back to the original "
                "values."
            ),
            "vgi.doc_md": (
                "**Cross-validated out-of-fold predictions** — no model is stored.\n\n"
                "- Each row is predicted by a model fit on the *other* folds (`cv :=`, default 5)\n"
                "- `estimator :=`, `target :=`, optional `id :=` passthrough, JSON `params :=`\n"
                "- Returns one `prediction` per input row (label or value), leakage-free\n\n"
                "Use it to make a held-out prediction column for metrics or model stacking."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM lightgbm.cross_val_predict("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM lightgbm.iris()), estimator := 'lgbm_classifier', target := 'target', id := 'sample_id')"
                ),
                description="5-fold out-of-fold predictions on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[CrossValArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("cross_val_predict requires 'target' (the label column name, e.g. target := 'label')")
        task, _ = build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        # The prediction column type depends on the target's label dtype; resolve
        # it from the input schema (string target -> VARCHAR, else BIGINT).
        classes: list[Any] | None = None
        if task == CLASSIFICATION and pa.types.is_string(input_schema.field(a.target).type):
            classes = [""]  # marker: string labels -> VARCHAR prediction column
        fields.append(_prediction_field(task, classes))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[CrossValArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[CrossValArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)
        task, _ = build_estimator(a.estimator, _parse_params(a.params))

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(
                pa.RecordBatch.from_pydict({n: [] for n in params.output_schema.names}, schema=params.output_schema)
            )
            return

        validate_features(table, feats)
        cat_mask = detect_categoricals(table, feats)
        categories = fit_categories(table, feats, cat_mask)
        cat_idx = categorical_indices(cat_mask)
        x = encode_matrix(table, feats, cat_mask, categories)
        hp = _parse_params(a.params)

        classes: list[Any] | None = None
        if task == CLASSIFICATION:
            y, classes = _target_codes(table, a.target)
        else:
            y = _target_array(table, a.target, task)

        preds = np.empty(len(y), dtype=float)
        for train_idx, test_idx in _cv_splitter(task, a.cv, y):
            est = _fit_fold(a.estimator, hp, x[train_idx], y[train_idx], cat_idx)
            preds[test_idx] = np.asarray(est.predict(x[test_idx])).reshape(-1)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = table.column(a.id).to_pylist()
        if task == CLASSIFICATION:
            assert classes is not None
            # estimator.predict returns the 0..n-1 code; decode to the original label.
            columns["prediction"] = [classes[int(round(v))] for v in preds]
        else:
            columns["prediction"] = [float(v) for v in preds]
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# cross_val_score (per-fold scores, no persistence)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class CrossValScoreArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[str, Arg("estimator", default="lgbm_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]


_CV_SCORE_SCHEMA = pa.schema(
    [
        sfield("fold", pa.int32(), "0-based fold index.", nullable=False),
        sfield("score", pa.float64(), "Held-out score for the fold (accuracy or R^2).", nullable=False),
        sfield("metric", pa.string(), "accuracy (classification) or r2 (regression).", nullable=False),
    ]
)


class CrossValScore(SinkBuffer[CrossValScoreArgs, DrainState]):
    FunctionArguments: ClassVar[type] = CrossValScoreArgs

    class Meta:
        name = "cross_val_score"
        description = "Per-fold cross-validation scores (accuracy or R^2); no model is stored"
        categories = ["models", "supervised", "evaluation"]
        tags = {
            "vgi.result_columns_md": columns_md(_CV_SCORE_SCHEMA),
            "vgi.doc_llm": (
                "Runs k-fold cross-validation and returns the held-out score for each fold (one `(fold, "
                "score, metric)` row), without storing any model. Name the label with `target :=`, choose "
                "the `estimator :=`, set the number of folds with `cv :=` (default 5), and tune via JSON "
                "`params :=`. The metric is the estimator's own scorer — accuracy for classifiers, R^2 for "
                "regressors. Aggregate the rows (e.g. `avg(score)`) for a single cross-validated "
                "performance estimate or to compare estimators/hyperparameters."
            ),
            "vgi.doc_md": (
                "**Cross-validated fold scores** — one row per fold, no model stored.\n\n"
                "- `estimator :=`, `target :=`, `cv :=` folds (default 5), JSON `params :=`\n"
                "- Returns `(fold, score, metric)` — the held-out score for each fold\n\n"
                "Scorer is the estimator's own (accuracy / R^2). Take `avg(score)` for a single "
                "cross-validated estimate."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT fold, score FROM lightgbm.cross_val_score("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM lightgbm.iris()), estimator := 'lgbm_classifier', target := 'target', cv := 5)"
                ),
                description="5-fold accuracy scores on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[CrossValScoreArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("cross_val_score requires 'target' (the label column name, e.g. target := 'label')")
        build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_CV_SCORE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[CrossValScoreArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[CrossValScoreArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)
        task, _ = build_estimator(a.estimator, _parse_params(a.params))

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("cross_val_score received no training rows")

        validate_features(table, feats)
        cat_mask = detect_categoricals(table, feats)
        categories = fit_categories(table, feats, cat_mask)
        cat_idx = categorical_indices(cat_mask)
        x = encode_matrix(table, feats, cat_mask, categories)
        y = _target_array(table, a.target, task)
        hp = _parse_params(a.params)
        metric = "accuracy" if task == CLASSIFICATION else "r2"

        folds: list[int] = []
        scores: list[float] = []
        for k, (train_idx, test_idx) in enumerate(_cv_splitter(task, a.cv, y)):
            est = _fit_fold(a.estimator, hp, x[train_idx], y[train_idx], cat_idx)
            folds.append(k)
            scores.append(float(est.score(x[test_idx], y[test_idx])))

        out.emit(
            pa.RecordBatch.from_pydict(
                {"fold": folds, "score": scores, "metric": [metric] * len(folds)},
                schema=params.output_schema,
            )
        )


# ===========================================================================
# Registry management: list_models / model_info / drop_model
# ===========================================================================

_MODEL_INFO_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Stored model name.", nullable=False),
        sfield("estimator", pa.string(), "Estimator type.", nullable=False),
        sfield("task", pa.string(), "classification or regression.", nullable=False),
        sfield("target", pa.string(), "Target column the model was trained on.", nullable=False),
        sfield("n_features", pa.int64(), "Number of features.", nullable=False),
        sfield("n_categorical", pa.int64(), "Number of categorical features.", nullable=False),
        sfield("n_samples", pa.int64(), "Number of training rows.", nullable=False),
        sfield("n_classes", pa.int64(), "Number of classes (NULL for regression)."),
        sfield("train_score", pa.float64(), "In-sample training score."),
        sfield("lightgbm_version", pa.string(), "lightgbm version used to fit."),
        sfield("created_at", pa.string(), "UTC timestamp the model was stored."),
        sfield("features", pa.list_(pa.string()), "Ordered feature column names.", nullable=False),
    ]
)


def _meta_rows(metas: list[ModelMetadata]) -> dict[str, list[Any]]:
    return {
        "model_name": [m.name for m in metas],
        "estimator": [m.estimator for m in metas],
        "task": [m.task for m in metas],
        "target": [m.target for m in metas],
        "n_features": [m.n_features for m in metas],
        "n_categorical": [sum(m.categorical) for m in metas],
        "n_samples": [m.n_samples for m in metas],
        "n_classes": [len(m.classes) if m.classes is not None else None for m in metas],
        "train_score": [m.train_score for m in metas],
        "lightgbm_version": [m.lightgbm_version for m in metas],
        "created_at": [m.created_at for m in metas],
        "features": [m.feature_names for m in metas],
    }


@dataclass(slots=True, frozen=True)
class NoArgs:
    pass


@init_single_worker
@bind_fixed_schema
class ListModels(TableFunctionGenerator[NoArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        name = "list_models"
        description = "List all models in the registry"
        categories = ["models", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_MODEL_INFO_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function listing every model persisted to the registry (by `fit` / "
                "`fit_lgbm_<task>` / search with a `model_name`). One row per model: `model_name`, "
                "`estimator`, `task`, `target`, training shape (`n_features`/`n_categorical`/`n_samples`/"
                "`n_classes`), `train_score`, `lightgbm_version`, `created_at`, and the ordered `features` "
                "list. Query it to discover what is available to `predict`/`explain`/`feature_importance`."
            ),
            "vgi.doc_md": (
                "**Model registry listing** — every saved model, one row each.\n\n"
                "- `model_name`, `estimator`, `task`, `target`\n"
                "- `n_features` / `n_categorical` / `n_samples` / `n_classes`, `train_score`\n"
                "- `lightgbm_version`, `created_at`, `features`\n\n"
                "Takes no arguments. Use it to find models to score with `predict`."
            ),
        }
        examples = [FunctionExample(sql="SELECT * FROM lightgbm.list_models()", description="List stored models")]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=10, max=10000)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        out.emit(pa.RecordBatch.from_pydict(_meta_rows(get_store().list()), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class ModelInfoArgs:
    model_name: Annotated[str, Arg(0, doc="Name of a stored model.")]


@init_single_worker
@bind_fixed_schema
class ModelInfo(TableFunctionGenerator[ModelInfoArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        name = "model_info"
        description = "Describe a single stored model (one row, empty if absent)"
        categories = ["models", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_MODEL_INFO_SCHEMA),
            "vgi.doc_llm": (
                "Returns the registry metadata for one model named positionally (`model_info('my_model')`): "
                "a single row with `model_name`, `estimator`, `task`, `target`, training shape "
                "(`n_features`/`n_categorical`/`n_samples`/`n_classes`), `train_score`, `lightgbm_version`, "
                "`created_at`, and the ordered `features` list. Emits zero rows if the model does not "
                "exist, so it never errors on a missing name. Use it to inspect a specific saved model "
                "before predicting."
            ),
            "vgi.doc_md": (
                "**Describe one stored model** — its registry metadata.\n\n"
                "- Call positionally: `model_info('my_model')`\n"
                "- One row: `estimator`, `task`, `target`, shape, `train_score`, `lightgbm_version`, "
                "`created_at`, `features`\n\n"
                "Returns no rows if the name is absent (never errors)."
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM lightgbm.model_info('iris_clf')", description="Show one model's metadata"
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[ModelInfoArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[ModelInfoArgs], state: None, out: OutputCollector) -> None:
        try:
            metas = [get_store().load_meta(params.args.model_name)]
        except ModelNotFoundError:
            metas = []
        out.emit(pa.RecordBatch.from_pydict(_meta_rows(metas), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class DropModelArgs:
    model_name: Annotated[str, Arg(0, doc="Name of the model to delete.")]


_DROP_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name of the model.", nullable=False),
        sfield("dropped", pa.bool_(), "True if a model was deleted, False if it did not exist.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class DropModel(TableFunctionGenerator[DropModelArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DROP_SCHEMA

    class Meta:
        name = "drop_model"
        description = "Delete a model from the registry"
        categories = ["models", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_DROP_SCHEMA),
            "vgi.doc_llm": (
                "Deletes a model from the registry by name (`drop_model('my_model')`) and returns one row "
                "`(model_name, dropped)` where `dropped` is true when a model was removed and false when "
                "the name did not exist. Idempotent and safe to call on an absent model. Use it to clean "
                "up models created by `fit`/`fit_lgbm_<task>`/search with a `model_name`."
            ),
            "vgi.doc_md": (
                "**Delete a stored model** — registry cleanup.\n\n"
                "- Call positionally: `drop_model('my_model')`\n"
                "- Returns `(model_name, dropped)`; `dropped` is false if the name did not exist\n\n"
                "Idempotent — safe on a missing model."
            ),
        }
        examples = [
            FunctionExample(sql="SELECT * FROM lightgbm.drop_model('iris_clf')", description="Delete a stored model")
        ]

    @classmethod
    def cardinality(cls, params: BindParams[DropModelArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[DropModelArgs], state: None, out: OutputCollector) -> None:
        name = params.args.model_name
        dropped = get_store().delete(name)
        out.emit(pa.RecordBatch.from_pydict({"model_name": [name], "dropped": [dropped]}, schema=params.output_schema))
        out.finish()


# Re-exported so importance.py and others can detect a malformed BLOB.
__all__ = ["ModelBlobError", "booster_from_text"]

MODEL_FUNCTIONS: list[type] = [
    FitModel,
    PredictModel,
    CrossValPredict,
    CrossValScore,
    ListModels,
    ModelInfo,
    DropModel,
]
