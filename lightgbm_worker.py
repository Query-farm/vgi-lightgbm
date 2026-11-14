# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.50.0",
#     "vgi-rpc[sentry]>=0.53.1",
#     "lightgbm>=4.0",
#     "scikit-learn>=1.5",
#     "numpy",
# ]
# ///
"""VGI worker exposing LightGBM to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_lightgbm`` into a single
``lightgbm`` catalog and runs the worker over stdio (local) or HTTP (Fly.io).

Usage:
    uv run lightgbm_worker.py           # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000         # serve over HTTP

    ATTACH 'lightgbm' (TYPE vgi, LOCATION 'uv run lightgbm_worker.py');
    SELECT * FROM lightgbm.main.fit((SELECT * FROM lightgbm.main.iris()), model_name := 'm', target := 'target');
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema, Table, constraint_record
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_lightgbm import __version__
from vgi_lightgbm.datasets import DATASET_FUNCTIONS
from vgi_lightgbm.importance import GAIN_VS_PERMUTATION_EXAMPLE, IMPORTANCE_FUNCTIONS, SHAP_BY_SPECIES_EXAMPLE
from vgi_lightgbm.models import MODEL_FUNCTIONS, OOF_CONFUSION_EXAMPLE
from vgi_lightgbm.search import SEARCH_FUNCTIONS
from vgi_lightgbm.typed_models import TYPED_FIT_FUNCTIONS

log = logging.getLogger(__name__)

DATA_VERSION = __version__
# data_version_spec is advertised as a SemVer *range* (a packaging SpecifierSet),
# not a bare version. The worker regenerates its data each release, so it serves
# exactly the current data version — an exact-match range.
DATA_VERSION_SPEC = f"=={DATA_VERSION}"
GIT_COMMIT = os.environ.get("VGI_LIGHTGBM_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by area.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *MODEL_FUNCTIONS,
    *TYPED_FIT_FUNCTIONS,
    *SEARCH_FUNCTIONS,
    *IMPORTANCE_FUNCTIONS,
]

# Provenance / about link advertised on the catalog (VGI source_url).
SOURCE_URL = "https://github.com/query-farm/vgi-lightgbm"

# Catalog-level metadata surfaced through duckdb_databases() (comment + tags).
# The description_llm/_md tags feed agent/doc consumers; author/copyright/license
# advertise provenance.
_CATALOG_COMMENT = "LightGBM train/predict model registry, datasets, and interpretation for DuckDB/SQL"
# Catalog-level description: the high-level "what this worker is".
_CATALOG_DESCRIPTION_LLM = (
    "LightGBM for SQL. Load datasets; fit gradient-boosted models (fit returns a "
    "model `BLOB`, predict aligns features by name and decodes string labels); run "
    "typed fit_<estimator>, grid and randomized hyperparameter search, feature "
    "importance, SHAP explanations, partial dependence, and permutation importance "
    "— all as DuckDB table functions."
)
_CATALOG_DESCRIPTION_MD = (
    "# LightGBM for SQL\n\n"
    "Exposes [LightGBM](https://lightgbm.readthedocs.io) to DuckDB/SQL as VGI functions:\n\n"
    "- **Datasets** — toy datasets and generators\n"
    "- **Models** — `fit`/`predict`, typed `fit_<estimator>`, cross-validation, "
    "grid/randomized search\n"
    "- **Interpretation** — feature importance, SHAP `explain`, partial dependence, "
    "permutation importance\n\n"
    "Models are stored as reusable BLOBs in a registry; native text serialization."
)
# Schema-level description: an index of what is callable in the `main` namespace.
_SCHEMA_DESCRIPTION_LLM = (
    "Functions in lightgbm.main, by family: datasets (table functions); models "
    "(fit → model `BLOB` → predict, typed fit_<estimator>, cross-validation, grid and "
    "randomized search); and interpretation (feature importance, SHAP explain, "
    "partial dependence, permutation importance). All are DuckDB table functions."
)
_SCHEMA_DESCRIPTION_MD = (
    "# `main` schema\n\n"
    "Every LightGBM function lives here, grouped by family:\n\n"
    "- **datasets** — toy/generated data as table functions\n"
    "- **models** — `fit`/`predict`, typed `fit_<estimator>`, cross-validation, "
    "grid/randomized search\n"
    "- **interpretation** — feature importance, SHAP `explain`, partial dependence, "
    "permutation importance\n\n"
    "Fit returns a reusable model `BLOB`; predict aligns features by name."
)
# Guaranteed-runnable, self-contained examples advertised on the catalog
# (VGI509): each is fully schema-qualified and executes as written against a
# freshly attached worker. Multi-statement examples run in order in one session,
# so a model can be fit, stashed in a session variable (the model `BLOB` can't ride
# a table function's single subquery slot), and then used by predict. Kept fast
# (small/default params, cv := 3) so each runs in well under ~30s.
_CATALOG_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Load the built-in iris dataset.",
            "sql": "SELECT * FROM lightgbm.main.iris LIMIT 5",
        },
        {
            "description": "Fit a LightGBM classifier on iris, then predict with the fitted model.",
            "sql": [
                (
                    "SET VARIABLE iris_model = ("
                    "SELECT model FROM lightgbm.main.fit((SELECT * EXCLUDE (target_name) FROM lightgbm.main.iris), "
                    "estimator := 'lgbm_classifier', target := 'target', id := 'sample_id'))"
                ),
                (
                    "SELECT sample_id, prediction "
                    "FROM lightgbm.main.predict((SELECT * FROM lightgbm.main.iris), "
                    "model := getvariable('iris_model'), id := 'sample_id') LIMIT 5"
                ),
            ],
        },
        {
            "description": "Three-fold cross-validated accuracy of a LightGBM classifier on iris.",
            "sql": (
                "SELECT fold, score FROM lightgbm.main.cross_val_score("
                "(SELECT * EXCLUDE (target_name) FROM lightgbm.main.iris), "
                "estimator := 'lgbm_classifier', target := 'target', cv := 3)"
            ),
        },
        {
            "description": "Grid-search a LightGBM classifier on iris over a tiny grid (3-fold).",
            "sql": (
                "SELECT params, mean_score, rank FROM lightgbm.main.grid_search("
                "(SELECT * EXCLUDE (target_name) FROM lightgbm.main.iris), "
                "target := 'target', cv := 3, "
                "estimator := 'lgbm_classifier', num_leaves := [15, 31], learning_rate := [0.1]) "
                "ORDER BY rank"
            ),
        },
        # Composed: out-of-fold predictions joined back to the dataset (shared constant).
        {"description": OOF_CONFUSION_EXAMPLE.description, "sql": OOF_CONFUSION_EXAMPLE.sql},
    ]
)
# The fixed agent-suitability suite (VGI152 / `vgi-lint simulate`). Each task is a
# question an analyst would actually ask; only `name` and `prompt` are published.
# The grader-only `reference_sql`/`check_sql` live in vgi-agent-tests.yaml (VGI416),
# which vgi-lint discovers beside pyproject.toml. Between them the tasks call every
# function the worker exposes, so a low score points at missing metadata rather
# than a missing test.
_CATALOG_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "iris_species_balance",
            "prompt": (
                "How many flowers of each species are in the built-in iris dataset? Return one row per species with"
                " columns named target_name and n, ordered by target_name."
            ),
        },
        {
            "name": "wine_sample_count",
            "prompt": (
                "How many wine samples are in the built-in wine-recognition dataset? Return a single row with one "
                "column named n."
            ),
        },
        {
            "name": "breast_cancer_cv_accuracy",
            "prompt": (
                "Before trusting a tumour-screening model I need an honest out-of-sample number. Using the built-in"
                " Wisconsin breast-cancer data and all 30 cell-nucleus features, run 5-fold cross-validation with a"
                " LightGBM classifier and report the mean held-out accuracy. Round to 4 decimals and return a "
                "single row with one column named mean_cv_accuracy."
            ),
        },
        {
            "name": "california_housing_priciest_latitude",
            "prompt": (
                "In the built-in California housing data, which one-degree latitude band (round the latitude to the"
                " nearest whole degree) has the highest average median house value? Return a single row with one "
                "column named lat."
            ),
        },
        {
            "name": "diabetes_train_rmse",
            "prompt": (
                "Fit a LightGBM regressor on the diabetes dataset to predict the one-year progression score from "
                "the baseline measurements, then score those same rows with the fitted model and report the root-"
                "mean-squared error between the predictions and the actual scores. Round to 2 decimals and return a"
                " single row with one column named train_rmse."
            ),
        },
        {
            "name": "iris_oof_accuracy",
            "prompt": (
                "I want a leakage-free accuracy for a LightGBM classifier on iris: predict every row from a model "
                "trained on the other folds (5 folds), then compare those out-of-fold predictions with the true "
                "labels. Round to 4 decimals and return a single row with one column named oof_accuracy."
            ),
        },
        {
            "name": "iris_grid_search_best_accuracy",
            "prompt": (
                "I'm tuning a LightGBM classifier on iris. Sweep num_leaves over 7, 15 and 31 and learning_rate "
                "over 0.05 and 0.1 with 5-fold cross-validation, and report the best mean cross-validated accuracy "
                "across all those combinations. Round to 4 decimals and return a single row with one column named "
                "best_cv_accuracy."
            ),
        },
        {
            "name": "iris_randomized_search_combos",
            "prompt": (
                "The full grid is too big to sweep, so sample it instead: on iris, try 4 randomly chosen "
                "combinations of num_leaves (7, 15, 31, 63) and learning_rate (0.03, 0.05, 0.1, 0.2) with 3-fold "
                "cross-validation, using 42 as the sampling seed. How many combinations end up on the leaderboard? "
                "Return a single row with one column named n_combos."
            ),
        },
        {
            "name": "fit_lgbm_classifier_summary",
            "prompt": (
                "Using the typed classifier fitter (the one that takes hyperparameters as named SQL arguments "
                "rather than JSON), train a LightGBM classifier on iris with target column 'target' and id column "
                "'sample_id', and report the training summary's sample and feature counts. Return one row with "
                "columns named n_samples and n_features."
            ),
        },
        {
            "name": "fit_lgbm_regressor_summary",
            "prompt": (
                "Using the typed regressor fitter, train a LightGBM regressor on the diabetes dataset (target "
                "column 'progression', id column 'sample_id') with 50 boosting rounds, and report the training "
                "summary's sample and feature counts. Return one row with columns named n_samples and n_features."
            ),
        },
        {
            "name": "diabetes_top_gain_feature",
            "prompt": (
                "Fit a LightGBM regressor on the diabetes dataset and tell me which single baseline measurement the"
                " booster relies on most, measured by total split gain. Return one row with a column named feature."
            ),
        },
        {
            "name": "diabetes_top_shap_feature",
            "prompt": (
                "Fit a LightGBM regressor on the diabetes dataset, then use per-row SHAP contributions to find the "
                "feature with the largest mean absolute contribution across all rows. Return one row with a column "
                "named feature."
            ),
        },
        {
            "name": "diabetes_top_permutation_feature",
            "prompt": (
                "Fit a LightGBM regressor on the diabetes dataset, then measure how much the model's score drops "
                "when each feature's values are shuffled (5 repeats, seed 42) and report the single most important "
                "feature by that measure. Return one row with a column named feature."
            ),
        },
        {
            "name": "bmi_partial_dependence_range",
            "prompt": (
                "Fit a LightGBM regressor on the diabetes dataset, then show how the model's average predicted "
                "progression moves as BMI varies over a 20-point grid, and report the lowest and highest values on "
                "that curve, each rounded to 2 decimals. Return one row with columns named min_pd and max_pd."
            ),
        },
        {
            "name": "registry_roundtrip",
            "prompt": (
                "Save a LightGBM classifier trained on iris under the name 'agent_iris_demo', confirm the registry "
                "can tell me which estimator and task it holds for that name, then delete it again. Return one row "
                "from the lookup with columns named estimator and task."
            ),
        },
        {
            "name": "registry_listing_after_drop",
            "prompt": (
                "Train a LightGBM classifier on iris and store it as 'agent_scratch_model', then delete that model "
                "from the registry and confirm it is gone: report how many models in the registry are named "
                "'agent_scratch_model' after the delete. Return a single row with one column named remaining."
            ),
        },
        {
            "name": "make_classification_class_balance",
            "prompt": (
                "Generate a synthetic classification dataset with 120 samples, 4 features, 3 classes and "
                "random_state 42, then report how many rows fall in each class. Return one row per class with "
                "columns named target and n, ordered by target."
            ),
        },
        {
            "name": "breast_cancer_oof_missed_malignant",
            "prompt": (
                "For a tumour screen the costly mistake is calling a malignant tumour benign. On the built-in "
                "Wisconsin breast-cancer data, using all 30 cell-nucleus features, get leakage-free 5-fold "
                "out-of-fold predictions from a LightGBM classifier and count how many tumours that are actually "
                "malignant were predicted benign. Return a single row with one column named missed_malignant."
            ),
        },
        {
            "name": "make_regression_spread",
            "prompt": (
                "Generate a synthetic regression dataset with 90 samples, 3 features, noise 5.0 and random_state "
                "42, then report how many rows it has and the standard deviation of its target rounded to 2 "
                "decimals. Return one row with columns named n and sd_target."
            ),
        },
    ]
)
_CATALOG_TAGS = {
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query Farm <hello@query.farm>",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": f"{SOURCE_URL}/issues",
    "vgi.support_policy_url": f"{SOURCE_URL}/blob/main/SUPPORT.md",
    "vgi.title": "LightGBM",
    "vgi.subtitle": (
        "Train, tune, evaluate, and explain gradient-boosted classifiers and regressors, with a saved-model registry."
    ),
    "vgi.keywords": json.dumps(
        [
            "lightgbm",
            "gradient boosting",
            "machine learning",
            "models",
            "classification",
            "regression",
            "cross-validation",
            "hyperparameter search",
            "feature importance",
            "shap",
        ]
    ),
    "vgi.executable_examples": _CATALOG_EXECUTABLE_EXAMPLES,
    "vgi.agent_test_tasks": _CATALOG_AGENT_TEST_TASKS,
}

# The `main` schema's navigation registry (vgi.categories, VGI413) and the
# primary category each object belongs to (vgi.category, VGI409/411/412). Array
# order is display order; every declared category must have at least one member.
_CATEGORY_REGISTRY = [
    {
        "name": "datasets",
        "title": "Datasets & Generators",
        "description": "Bundled reference datasets and synthetic data generators to train on.",
        "keywords": ["datasets", "sample data", "generators", "toy dataset"],
    },
    {
        "name": "training",
        "title": "Training",
        "description": "Fit a LightGBM model and get back a reusable model `BLOB`.",
        "keywords": ["fit", "train", "gradient boosting", "supervised"],
    },
    {
        "name": "inference",
        "title": "Inference",
        "description": "Score new rows through an already-fit model.",
        "keywords": ["predict", "score", "inference"],
    },
    {
        "name": "evaluation",
        "title": "Evaluation",
        "description": "Cross-validated scores and out-of-fold predictions for honest model assessment.",
        "keywords": ["cross-validation", "evaluation", "scoring"],
    },
    {
        "name": "tuning",
        "title": "Hyperparameter Search",
        "description": "Grid and randomized search over a typed hyperparameter space.",
        "keywords": ["grid search", "randomized search", "hyperparameters", "tuning"],
    },
    {
        "name": "interpretation",
        "title": "Interpretation",
        "description": "Explain what a fitted model learned: importances, SHAP values, partial dependence.",
        "keywords": ["feature importance", "shap", "explainability", "partial dependence"],
    },
    {
        "name": "registry",
        "title": "Model Registry",
        "description": "Discover, inspect, and delete the models persisted by name.",
        "keywords": ["registry", "model management", "metadata"],
    },
]
# Object name -> primary category. Every table and function must appear here.
_CATEGORY_BY_NAME = {
    "iris": "datasets",
    "wine": "datasets",
    "breast_cancer": "datasets",
    "diabetes": "datasets",
    "california_housing": "datasets",
    "make_classification": "datasets",
    "make_regression": "datasets",
    "fit": "training",
    "fit_lgbm_classifier": "training",
    "fit_lgbm_regressor": "training",
    "predict": "inference",
    "cross_val_predict": "evaluation",
    "cross_val_score": "evaluation",
    "grid_search": "tuning",
    "randomized_search": "tuning",
    "feature_importance": "interpretation",
    "explain": "interpretation",
    "permutation_importance": "interpretation",
    "partial_dependence": "interpretation",
    "models": "registry",
    "model_info": "registry",
    "drop_model": "registry",
}


# Per-schema metadata for the single `main` schema (VGI506/124/126): title,
# keywords (JSON array), and a runnable, schema-qualified example query.
_SCHEMA_TAGS = {
    "provider": "LightGBM",
    "domain": "machine-learning",
    "vgi.title": "LightGBM",
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    "vgi.keywords": json.dumps(
        [
            "lightgbm",
            "gradient boosting",
            "models",
            "datasets",
            "interpretation",
            "cross-validation",
        ]
    ),
    "vgi.categories": json.dumps(_CATEGORY_REGISTRY),
    "vgi.example_queries": json.dumps(
        [
            {
                "description": "Fit a LightGBM classifier on iris and return the training summary",
                "sql": (
                    "SELECT estimator, task, n_samples, n_features, train_score "
                    "FROM lightgbm.main.fit((SELECT * EXCLUDE (target_name) FROM lightgbm.main.iris), "
                    "estimator := 'lgbm_classifier', target := 'target', id := 'sample_id')"
                ),
            },
            # Composed examples, each defined once beside the function that drives it.
            *(
                {"description": e.description, "sql": e.sql}
                for e in (OOF_CONFUSION_EXAMPLE, GAIN_VS_PERMUTATION_EXAMPLE, SHAP_BY_SPECIES_EXAMPLE)
            ),
        ]
    ),
}


def _humanize(name: str) -> str:
    """Title-case a snake_case function name for a display title."""
    return name.replace("_", " ").title()


def _apply_discovery_tags(functions: list[type]) -> None:
    """Inject the per-function discovery tags the catalog-quality linter expects.

    ``vgi.title`` and ``vgi.keywords`` (a JSON array of strings) are derived
    mechanically from each function's existing Meta (display name, categories).
    ``vgi.source_url`` is deliberately NOT set here — it is a catalog-only tag
    (VGI139). The richer ``vgi.doc_llm`` / ``vgi.doc_md`` tags are authored per
    function in the implementation modules and are left untouched here.

    ``vgi.example_queries`` is mirrored from each function's ``Meta.examples``.
    The native ``duckdb_functions().examples`` column the extension populates is a
    bare ``VARCHAR[]`` of SQL, so a ``FunctionExample``'s ``description`` never
    reaches a consumer through it; re-publishing the same examples as the
    structured tag is what carries the descriptions (VGI515). The two carriers are
    deduped by SQL downstream, so nothing runs twice.

    ``vgi.category`` is the object's single primary navigation category, taken
    from ``_CATEGORY_BY_NAME`` — the join key into the schema's ``vgi.categories``
    registry (VGI409/VGI411/VGI412).
    """
    for fn in functions:
        meta = getattr(fn, "Meta", None)
        if meta is None:
            continue
        name = getattr(meta, "name", fn.__name__)
        cats = list(getattr(meta, "categories", []) or [])
        tags = dict(getattr(meta, "tags", {}) or {})
        tags.setdefault("vgi.title", _humanize(name))
        keywords = list(dict.fromkeys(cats or name.split("_")))
        tags.setdefault("vgi.keywords", json.dumps(keywords))
        tags.setdefault("vgi.category", _CATEGORY_BY_NAME[name])
        examples = list(getattr(meta, "examples", []) or [])
        if examples:
            tags.setdefault(
                "vgi.example_queries",
                json.dumps(
                    [{"description": e.description, "sql": e.sql} for e in examples],
                ),
            )
        meta.tags = tags


_apply_discovery_tags(_FUNCTIONS)


def _is_parameterless_table_fn(fn: type) -> bool:
    """True for a table function whose argument dataclass has no fields.

    Such a function always returns the same rows, so it is also exposed as a
    plain table (VGI311) — ``SELECT * FROM schema.name`` without parentheses.
    """
    args = getattr(fn, "FunctionArguments", None)
    return args is not None and dataclasses.is_dataclass(args) and not dataclasses.fields(args)


# Table-specific metadata for the parameterless functions also exposed as tables
# (VGI311). Each carries a table-oriented description, a descriptive title (not a
# restatement of the name), a primary key, and a runnable example — distinct from
# the backing function's documentation.
_TABLE_META: dict[str, dict[str, Any]] = {
    "iris": {
        "title": "Fisher's iris flowers",
        "comment": "150 iris flowers with four sepal/petal measurements (cm) and their species.",
        "keywords": json.dumps(["iris", "flowers", "classification", "toy dataset", "fisher"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Fisher's classic iris table: 150 rows, one per flower. Columns are `sample_id`, four numeric "
            "measurements in centimetres (`sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, "
            "`petal_width_cm`), an integer `target` (0/1/2), and the `target_name` species "
            "(setosa/versicolor/virginica). Reach for it whenever you need a small, balanced 3-class "
            "toy dataset to feed `fit`/`predict` — the table is queryable directly, no arguments."
        ),
        "doc_md": (
            "### `iris` table\n\n"
            "150 iris flowers, evenly split across three species:\n\n"
            "- `sample_id` — row id\n"
            "- four measurements in **cm** — `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, "
            "`petal_width_cm`\n"
            "- `target` (0–2) and `target_name` (species)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Row count per species",
                    "sql": "SELECT target_name, count(*) FROM lightgbm.main.iris GROUP BY target_name",
                }
            ]
        ),
    },
    "wine": {
        "title": "Wine cultivar chemistry",
        "comment": "178 wines with 13 chemical measurements and their cultivar of origin.",
        "keywords": json.dumps(["wine", "chemistry", "classification", "toy dataset", "cultivars"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "The wine-recognition table: 178 rows, one per wine sample, with 13 numeric chemical-analysis "
            "features (alcohol, malic acid, ash, magnesium, total phenols, flavanoids, colour intensity, "
            "hue, proline, ...), an integer `target`, and the cultivar `target_name`. A 3-class dataset "
            "for classification over continuous features."
        ),
        "doc_md": (
            "### `wine` table\n\n"
            "178 wines from three cultivars:\n\n"
            "- `sample_id` — row id\n"
            "- 13 chemical features (alcohol, phenols, colour intensity, proline, ...)\n"
            "- `target` (0–2) and `target_name` (cultivar)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Average alcohol per cultivar",
                    "sql": "SELECT target_name, round(avg(alcohol), 2) FROM lightgbm.main.wine GROUP BY target_name",
                }
            ]
        ),
    },
    "breast_cancer": {
        "title": "Breast-cancer cell diagnostics",
        "comment": "569 tumour samples with 30 cell-nucleus measurements and a benign/malignant label.",
        "keywords": json.dumps(["breast cancer", "diagnostics", "classification", "toy dataset", "binary"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "The Wisconsin breast-cancer table: 569 rows, one per tumour, with 30 numeric features "
            "summarising cell-nucleus geometry (mean/standard-error/worst of radius, texture, area, "
            "concavity, ...), a binary `target` (0 = malignant, 1 = benign), and `target_name`. A standard "
            "binary-classification benchmark."
        ),
        "doc_md": (
            "### `breast_cancer` table\n\n"
            "569 tumour samples, binary outcome:\n\n"
            "- `sample_id` — row id\n"
            "- 30 cell-nucleus features (mean / SE / worst of radius, texture, area, ...)\n"
            "- `target` (0 malignant, 1 benign) and `target_name`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Class balance",
                    "sql": "SELECT target_name, count(*) FROM lightgbm.main.breast_cancer GROUP BY target_name",
                }
            ]
        ),
    },
    "diabetes": {
        "title": "Diabetes progression",
        "comment": "442 patients with 10 baseline measurements and a one-year disease-progression score.",
        "keywords": json.dumps(["diabetes", "regression", "health", "toy dataset", "progression"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Diabetes table: 442 rows, one per patient, with 10 mean-centred, scaled baseline features "
            "(`age`, `sex`, `bmi`, blood pressure `bp`, and six serum measurements `s1`..`s6`) and a "
            "continuous `target` quantifying disease progression one year after baseline. A small "
            "regression dataset — handy for `explain` and `partial_dependence`."
        ),
        "doc_md": (
            "### `diabetes` table\n\n"
            "442 patients (regression):\n\n"
            "- `sample_id` — row id\n"
            "- 10 scaled baseline features (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`)\n"
            "- `target` — one-year disease-progression score"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Range and mean of the one-year progression score",
                    "sql": (
                        "SELECT min(progression) AS lo, max(progression) AS hi, "
                        "round(avg(progression), 1) AS mean FROM lightgbm.main.diabetes"
                    ),
                }
            ]
        ),
    },
    "california_housing": {
        "title": "California housing prices",
        "comment": "20640 census block groups with 8 features and the median house value (regression).",
        "keywords": json.dumps(["california housing", "regression", "prices", "toy dataset", "census"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "California-housing table: 20,640 rows, one per 1990 census block group, with 8 numeric "
            "features (`medinc` median income, `houseage`, average rooms/bedrooms, population, occupancy, "
            "latitude, longitude) and a continuous `target` — the median house value in $100,000s. A large "
            "regression dataset; downloaded from scikit-learn on first use."
        ),
        "doc_md": (
            "### `california_housing` table\n\n"
            "20,640 census block groups (regression):\n\n"
            "- `sample_id` — row id\n"
            "- 8 features (`medinc`, `houseage`, `averooms`, `latitude`, `longitude`, ...)\n"
            "- `target` — median house value (in $100k)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Average median house value by latitude band",
                    "sql": (
                        "SELECT round(latitude) AS lat, round(avg(median_house_value), 2) AS avg_value "
                        "FROM lightgbm.main.california_housing GROUP BY lat ORDER BY lat"
                    ),
                }
            ]
        ),
    },
    "models": {
        "title": "Saved model registry",
        "comment": "One row per model persisted in the registry, with its estimator, task, shape, and score.",
        "keywords": json.dumps(["model registry", "models", "metadata", "catalog", "persistence"]),
        "primary_key": (("model_name",),),
        "doc_llm": (
            "A table of every model saved to the registry (by `fit`/`fit_lgbm_<task>`/search with a "
            "`model_name`). One row per model keyed by `model_name`, with its `estimator`, `task`, "
            "`target`, training shape (`n_features`/`n_categorical`/`n_samples`/`n_classes`), "
            "`train_score`, `lightgbm_version`, `created_at`, and the `features` list. Query it to discover "
            "what is available to `predict`/`explain`/`feature_importance`."
        ),
        "doc_md": (
            "### `models` table\n\n"
            "Every model in the registry, one row each:\n\n"
            "- `model_name` (key), `estimator`, `task`, `target`\n"
            "- `n_features` / `n_categorical` / `n_samples` / `n_classes`, `train_score`\n"
            "- `lightgbm_version`, `created_at`, `features`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "How many models are saved",
                    "sql": "SELECT count(*) AS model_count FROM lightgbm.main.models",
                }
            ]
        ),
    },
}


def _function_table(fn: type) -> Table:
    """Expose a parameterless table function as a same-named table (VGI311).

    The table carries its own table-oriented metadata from ``_TABLE_META`` — a
    description, descriptive title, primary key, and example — distinct from the
    backing function's documentation.
    """
    meta = getattr(fn, "Meta")  # noqa: B009 - Meta is a dynamic per-function class
    tm = _TABLE_META[meta.name]
    primary_key = tm["primary_key"]
    # The key columns are inherently non-null (VGI804).
    not_null = tuple(dict.fromkeys(col for cols in primary_key for col in cols))
    # vgi.v3 carries constraints only as canonical records. The worker produces
    # every row of these read-only tables, so the key holds for all of them.
    constraints = (
        *(
            constraint_record(
                ["main"],
                meta.name,
                f"pk_{i}" if i else "pk",
                "PRIMARY_KEY",
                [[col] for col in cols],
                enforcement="ENFORCED",
                validation="ALL_ROWS",
            )
            for i, cols in enumerate(primary_key)
        ),
        *(
            constraint_record(
                ["main"],
                meta.name,
                f"{col}_not_null",
                "NOT_NULL",
                [[col]],
                enforcement="ENFORCED",
                validation="ALL_ROWS",
            )
            for col in not_null
        ),
    )
    return Table(
        name=meta.name,
        function=fn,
        comment=tm["comment"],
        constraints=constraints,
        tags={
            "provider": "LightGBM",
            "domain": "machine-learning",
            "vgi.title": tm["title"],
            "vgi.keywords": tm["keywords"],
            "vgi.doc_llm": tm["doc_llm"],
            "vgi.doc_md": tm["doc_md"],
            "vgi.example_queries": tm["example_queries"],
            "vgi.category": _CATEGORY_BY_NAME[meta.name],
        },
    )


_TABLES = [_function_table(fn) for fn in _FUNCTIONS if _is_parameterless_table_fn(fn)]

_LIGHTGBM_CATALOG = Catalog(
    name="lightgbm",
    default_schema="main",
    comment=_CATALOG_COMMENT,
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            path=["main"],
            comment="LightGBM train/predict model registry, datasets, and interpretation for SQL",
            tags=_SCHEMA_TAGS,
            tables=_TABLES,
            functions=list(_FUNCTIONS),
        ),
    ],
)


class LightGBMCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _LIGHTGBM_CATALOG
    catalog_name = _LIGHTGBM_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=GIT_COMMIT,
                data_version_spec=DATA_VERSION_SPEC,
                source_url=SOURCE_URL,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=GIT_COMMIT,
        )


class LightGBMWorker(Worker):
    """Worker process hosting the LightGBM catalog."""

    catalog = _LIGHTGBM_CATALOG
    catalog_interface = LightGBMCatalog


def main() -> None:
    """Run the LightGBM worker process (stdio or, via flags, HTTP)."""
    LightGBMWorker.main()


if __name__ == "__main__":
    main()
