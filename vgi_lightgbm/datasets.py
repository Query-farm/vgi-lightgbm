"""Reference datasets exposed as DuckDB table functions.

LightGBM ships no datasets of its own, so these reuse scikit-learn's bundled
data purely to keep demos and the SQL tests self-contained:

* **Toy datasets** -- ``iris()``, ``wine()``, ``breast_cancer()``
  (classification) and ``diabetes()`` (regression). Zero-argument, fixed schema.
* **Synthetic generators** -- ``make_classification()`` / ``make_regression()``.
  Their column count depends on arguments, so they build their schema in
  ``on_bind``.

    SELECT * FROM lightgbm.main.iris();
    SELECT * FROM lightgbm.main.make_classification(n_samples := 500, n_features := 8);
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from sklearn import datasets as skd
from vgi.arguments import Arg
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from .schema_utils import (
    NoArgs,
    dedupe_names,
    field,
    result_columns_schema,
    result_dynamic_columns_md,
    snake_case,
)

_RESERVED = {"sample_id", "target", "target_name", "progression", "median_house_value"}


def _feature_labels(bunch: Any, n_features: int) -> list[str]:
    """scikit-learn feature names if present and well-sized, else ``feature_{i}``."""
    names = getattr(bunch, "feature_names", None)
    if names is not None and len(names) == n_features:
        return [str(n) for n in names]
    return [f"feature_{i}" for i in range(n_features)]


# Per-column definitions for the bundled datasets. A numeric column's comment has
# to say what the number *means* — its unit or its definition — not just repeat
# the column name (VGI131), and these datasets' features are only meaningful with
# that context (the diabetes columns, for instance, are scaled, not raw).
_WINE_FEATURE_DOCS: dict[str, str] = {
    "alcohol": "Alcohol content, in percent by volume.",
    "malic_acid": "Malic acid concentration, in grams per litre.",
    "ash": "Total ash (mineral residue after incineration), in grams per litre.",
    "alcalinity_of_ash": "Alkalinity of the ash, in milliequivalents per litre.",
    "magnesium": "Magnesium concentration, in milligrams per litre.",
    "total_phenols": "Total phenol content, as a concentration index.",
    "flavanoids": "Flavanoid phenol content, as a concentration index.",
    "nonflavanoid_phenols": "Non-flavanoid phenol content, as a concentration index.",
    "proanthocyanins": "Proanthocyanin content, as a concentration index.",
    "color_intensity": "Colour intensity of the wine, as an absorbance index.",
    "hue": "Colour hue, a ratio derived from the absorbance spectrum.",
    "od280_od315_of_diluted_wines": (
        "Ratio of absorbance at 280 nm to absorbance at 315 nm for the diluted wine (a protein-content index)."
    ),
    "proline": "Proline (an amino acid) concentration, in milligrams per litre.",
}

# The Wisconsin diagnostic features are ten geometry/texture measurements of the
# cell nuclei in a digitised image, each reported three ways.
_CANCER_BASE_DOCS: dict[str, str] = {
    "radius": "nucleus radius (mean distance from the centre to the perimeter), in image pixels",
    "texture": "nucleus texture, the standard deviation of the grey-scale pixel values",
    "perimeter": "nucleus perimeter length, in image pixels",
    "area": "nucleus area, in square image pixels",
    "smoothness": "nucleus smoothness, the local variation in radius lengths (a ratio)",
    "compactness": "nucleus compactness, perimeter squared over area minus one (a ratio)",
    "concavity": "severity of the concave portions of the nucleus contour (a ratio)",
    "concave_points": "count of concave portions of the nucleus contour",
    "symmetry": "nucleus symmetry, as a ratio",
    "fractal_dimension": "nucleus fractal dimension ('coastline approximation' minus one, a ratio)",
}
_CANCER_STAT_DOCS: dict[str, str] = {
    "mean": "Mean over the image's nuclei of the {base}.",
    "error": "Standard error over the image's nuclei of the {base}.",
    "worst": "Largest ('worst') value over the image's nuclei of the {base}.",
}

_DIABETES_FEATURE_DOCS: dict[str, str] = {
    "age": "Patient age at baseline",
    "sex": "Patient sex, encoded numerically",
    "bmi": "Body mass index at baseline",
    "bp": "Average blood pressure at baseline",
    "s1": "Serum measurement 1: total serum cholesterol (tc)",
    "s2": "Serum measurement 2: low-density lipoproteins (ldl)",
    "s3": "Serum measurement 3: high-density lipoproteins (hdl)",
    "s4": "Serum measurement 4: total cholesterol over HDL (tch)",
    "s5": "Serum measurement 5: log of serum triglycerides (ltg)",
    "s6": "Serum measurement 6: blood sugar level (glu)",
}
# Every diabetes feature is mean-centred and scaled by (n_samples * stddev), so
# each column has unit L2 norm and is dimensionless -- the raw units are gone.
_DIABETES_SCALING = "mean-centred and scaled to unit L2 norm across the 442 patients (dimensionless)"

_CALIFORNIA_FEATURE_DOCS: dict[str, str] = {
    "medinc": "Median income of the block group's households, in tens of thousands of US dollars.",
    "houseage": "Median age of the houses in the block group, in years.",
    "averooms": "Average number of rooms per household in the block group (a count per household).",
    "avebedrms": "Average number of bedrooms per household in the block group (a count per household).",
    "population": "Total count of people resident in the block group.",
    "aveoccup": "Average household size, in people per household.",
    "latitude": "Latitude of the block group's centroid, in decimal degrees north.",
    "longitude": "Longitude of the block group's centroid, in decimal degrees east (negative west).",
}


def _cancer_feature_docs() -> dict[str, str]:
    """Expand the ten base measurements into the 30 mean/error/worst columns."""
    docs: dict[str, str] = {}
    for base, text in _CANCER_BASE_DOCS.items():
        docs[f"mean_{base}"] = _CANCER_STAT_DOCS["mean"].format(base=text)
        docs[f"{base}_error"] = _CANCER_STAT_DOCS["error"].format(base=text)
        docs[f"worst_{base}"] = _CANCER_STAT_DOCS["worst"].format(base=text)
    return docs


def _diabetes_feature_docs() -> dict[str, str]:
    """Every diabetes feature, with the shared scaling stated on each column."""
    return {name: f"{text}, {_DIABETES_SCALING}." for name, text in _DIABETES_FEATURE_DOCS.items()}


def _feature_fields(labels: list[str], docs: dict[str, str] | None = None) -> list[pa.Field]:
    """Feature columns, each documented with its unit/definition where known.

    ``docs`` maps the sanitised column name to its description; a label that
    already carries its unit (scikit-learn's iris labels are ``"... (cm)"``) falls
    back to the label itself.
    """
    cols = dedupe_names([snake_case(label) for label in labels])
    docs = docs or {}
    return [
        field(col, pa.float64(), docs.get(col) or f"Feature: {label}.", nullable=False)
        for col, label in zip(cols, labels, strict=True)
    ]


def _classification_schema(labels: list[str], target_names: list[str], docs: dict[str, str] | None = None) -> pa.Schema:
    """id + float features + integer target + human-readable target name."""
    fields = [field("sample_id", pa.int32(), "Row index within the dataset (0-based).", nullable=False)]
    fields.extend(_feature_fields(labels, docs))
    fields.append(field("target", pa.int32(), "Integer class label (0-based).", nullable=False))
    fields.append(
        field(
            "target_name",
            pa.dictionary(pa.int8(), pa.string()),
            f"Human-readable class name (one of: {', '.join(target_names)}).",
            nullable=False,
        )
    )
    return pa.schema(fields)


def _regression_schema(
    labels: list[str], target_col: str, target_doc: str, docs: dict[str, str] | None = None
) -> pa.Schema:
    """id + float features + a continuous, domain-named float target.

    Regression targets get a domain name (``progression``, ``median_house_value``)
    rather than a bare ``target``: the classification datasets already use
    ``target`` for their integer class label, and one column name must map to one
    SQL type across the catalog (VGI205).
    """
    fields = [field("sample_id", pa.int32(), "Row index within the dataset (0-based).", nullable=False)]
    fields.extend(_feature_fields(labels, docs))
    fields.append(field(target_col, pa.float64(), target_doc, nullable=False))
    return pa.schema(fields)


def _synthetic_schema(n_features: int, target_col: str, target_type: pa.DataType, target_doc: str) -> pa.Schema:
    fields = [field("sample_id", pa.int32(), "Row index within the generated sample (0-based).", nullable=False)]
    fields.extend(_feature_fields([f"feature_{i}" for i in range(n_features)]))
    fields.append(field(target_col, target_type, target_doc, nullable=False))
    return pa.schema(fields)


def _emit_matrix(
    data: Any,
    targets: dict[str, list[Any]],
    schema: pa.Schema,
    out: OutputCollector,
    output_schema: pa.Schema,
) -> None:
    """Emit a feature matrix + named target column(s) as one record batch."""
    n_rows, _ = data.shape
    feature_cols = [name for name in schema.names if name not in _RESERVED]
    columns: dict[str, Any] = {"sample_id": list(range(n_rows))}
    for j, col in enumerate(feature_cols):
        columns[col] = data[:, j].tolist()
    columns.update(targets)
    out.emit(pa.RecordBatch.from_pydict(columns, schema=output_schema))
    out.finish()


# ===========================================================================
# Toy datasets
# ===========================================================================


class _ToyDataset(TableFunctionGenerator[NoArgs]):
    """Base for fixed-schema toy datasets. Subclasses set BUNCH + REGRESSION + Meta."""

    BUNCH: ClassVar[Any]
    REGRESSION: ClassVar[bool] = False
    # Name of the emitted target column (regression sets a domain-specific one).
    TARGET_COLUMN: ClassVar[str] = "target"
    FIXED_SCHEMA: ClassVar[pa.Schema]  # set on each subclass via @bind_fixed_schema

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        n = int(cls.BUNCH.data.shape[0])
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        bunch = cls.BUNCH
        target = bunch.target
        targets: dict[str, list[Any]]
        if cls.REGRESSION:
            targets = {cls.TARGET_COLUMN: [float(t) for t in target]}
        else:
            names = list(bunch.target_names)
            targets = {
                "target": [int(t) for t in target],
                "target_name": [str(names[int(t)]) for t in target],
            }
        _emit_matrix(bunch.data, targets, cls.FIXED_SCHEMA, out, params.output_schema)


def _classification_toy(loader: Any, docs: dict[str, str] | None = None) -> tuple[Any, pa.Schema]:
    bunch = loader()
    labels = _feature_labels(bunch, bunch.data.shape[1])
    return bunch, _classification_schema(labels, [str(n) for n in bunch.target_names], docs)


def _regression_toy(
    loader: Any, target_col: str, target_doc: str, docs: dict[str, str] | None = None
) -> tuple[Any, pa.Schema]:
    bunch = loader()
    labels = _feature_labels(bunch, bunch.data.shape[1])
    return bunch, _regression_schema(labels, target_col, target_doc, docs)


_IRIS, _IRIS_SCHEMA = _classification_toy(skd.load_iris)
_WINE, _WINE_SCHEMA = _classification_toy(skd.load_wine, _WINE_FEATURE_DOCS)
_CANCER, _CANCER_SCHEMA = _classification_toy(skd.load_breast_cancer, _cancer_feature_docs())
_DIABETES, _DIABETES_SCHEMA = _regression_toy(
    skd.load_diabetes,
    "progression",
    "Quantitative measure of disease progression one year after baseline (unitless index, ~25-346).",
    _diabetes_feature_docs(),
)


@init_single_worker
@bind_fixed_schema
class IrisFunction(_ToyDataset):
    """Fisher's iris dataset: 150 flowers, 4 measurements, 3 species."""

    BUNCH = _IRIS
    FIXED_SCHEMA: ClassVar[pa.Schema] = _IRIS_SCHEMA

    class Meta:
        name = "iris"
        description = "Fisher's iris dataset (150 samples, 4 features, 3 species)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_IRIS_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning Fisher's classic iris dataset: 150 rows, four "
                "numeric flower measurements in centimetres (`sepal_length_cm`, `sepal_width_cm`, "
                "`petal_length_cm`, `petal_width_cm`), an integer `target` (0/1/2), a human-readable "
                "`target_name` species, and a 0-based `sample_id`. It is the default demo input for nearly "
                "every other function in this worker (fit/predict, search, importance). Use it as a small, "
                "well-separated 3-class benchmark or to smoke-test a LightGBM pipeline end to end."
            ),
            "vgi.doc_md": (
                "**Iris dataset** — 150 flowers, 4 measurements, 3 species (the canonical ML toy set).\n\n"
                "- `sample_id` `INTEGER` — 0-based row index\n"
                "- four `*_cm` `DOUBLE` feature columns (sepal/petal length & width)\n"
                "- `target` `INTEGER` (0–2) and `target_name` the species name\n\n"
                "Takes no arguments. The go-to small, balanced 3-class classification demo and the input "
                "used in most examples here."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT target_name, count(*) AS n, round(avg(petal_length_cm), 2) AS avg_petal_length_cm "
                    "FROM lightgbm.main.iris() GROUP BY target_name ORDER BY target_name"
                ),
                description="Row count and mean petal length per species",
            ),
            FunctionExample(
                sql="SELECT target_name, avg(petal_length_cm) FROM lightgbm.main.iris() GROUP BY target_name",
                description="Mean petal length per species",
            ),
        ]


@init_single_worker
@bind_fixed_schema
class WineFunction(_ToyDataset):
    """Wine recognition dataset: 178 samples, 13 chemical features, 3 cultivars."""

    BUNCH = _WINE
    FIXED_SCHEMA: ClassVar[pa.Schema] = _WINE_SCHEMA

    class Meta:
        name = "wine"
        description = "Wine recognition dataset (178 samples, 13 features, 3 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_WINE_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the wine-recognition dataset: 178 rows, one per "
                "wine sample, with 13 numeric chemical-analysis features (alcohol, malic acid, ash, "
                "magnesium, total phenols, flavanoids, colour intensity, hue, proline, ...), an integer "
                "`target` (0/1/2), the cultivar `target_name`, and a `sample_id`. A balanced 3-class "
                "dataset over continuous features — handy for multiclass classification demos."
            ),
            "vgi.doc_md": (
                "**Wine dataset** — 178 wines from three cultivars.\n\n"
                "- `sample_id` `INTEGER` — 0-based row index\n"
                "- 13 `DOUBLE` chemical features (alcohol, phenols, colour intensity, proline, ...)\n"
                "- `target` `INTEGER` (0–2) and `target_name` the cultivar\n\n"
                "Takes no arguments. A continuous-feature 3-class classification benchmark."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sample_id, alcohol, flavanoids, target_name "
                    "FROM lightgbm.main.wine() WHERE alcohol > 13.5 ORDER BY alcohol DESC LIMIT 10"
                ),
                description="The ten strongest wines with their cultivar",
            )
        ]


@init_single_worker
@bind_fixed_schema
class BreastCancerFunction(_ToyDataset):
    """Breast cancer Wisconsin: 569 samples, 30 features, 2 classes."""

    BUNCH = _CANCER
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CANCER_SCHEMA

    class Meta:
        name = "breast_cancer"
        description = "Breast cancer Wisconsin diagnostic (569 samples, 30 features, 2 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_CANCER_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the Wisconsin breast-cancer diagnostic dataset: "
                "569 rows, one per tumour, with 30 numeric features summarising cell-nucleus geometry "
                "(mean / standard-error / worst of radius, texture, perimeter, area, concavity, ...), a "
                "binary `target` (0 = malignant, 1 = benign), `target_name`, and a `sample_id`. A standard "
                "binary-classification benchmark for fit/predict and probability/SHAP demos."
            ),
            "vgi.doc_md": (
                "**Breast cancer (Wisconsin) dataset** — 569 tumour samples, binary outcome.\n\n"
                "- `sample_id` `INTEGER` — 0-based row index\n"
                "- 30 `DOUBLE` cell-nucleus features (mean / SE / worst of radius, texture, area, ...)\n"
                "- `target` `INTEGER` (0 malignant, 1 benign) and `target_name`\n\n"
                "Takes no arguments. The canonical binary-classification toy set."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT target_name, count(*) AS n, round(avg(mean_radius), 2) AS avg_mean_radius "
                    "FROM lightgbm.main.breast_cancer() GROUP BY target_name ORDER BY target_name"
                ),
                description="Sample count and mean nucleus radius per diagnosis",
            )
        ]


@init_single_worker
@bind_fixed_schema
class DiabetesFunction(_ToyDataset):
    """Diabetes regression: 442 samples, 10 baseline features, continuous target."""

    BUNCH = _DIABETES
    REGRESSION = True
    TARGET_COLUMN: ClassVar[str] = "progression"
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DIABETES_SCHEMA

    class Meta:
        name = "diabetes"
        description = "Diabetes progression regression (442 samples, 10 features)"
        categories = ["datasets", "regression"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_DIABETES_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the diabetes regression dataset: 442 rows, one per "
                "patient, with 10 mean-centred, scaled baseline features (`age`, `sex`, `bmi`, blood "
                "pressure `bp`, and six serum measurements `s1`..`s6`), a continuous `target` quantifying "
                "disease progression one year after baseline, and a `sample_id`. A small regression "
                "benchmark — the default demo input for `explain` and `partial_dependence`."
            ),
            "vgi.doc_md": (
                "**Diabetes dataset** — 442 patients, continuous regression target.\n\n"
                "- `sample_id` `INTEGER` — 0-based row index\n"
                "- 10 `DOUBLE` scaled baseline features (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`)\n"
                "- `target` `DOUBLE` — one-year disease-progression score\n\n"
                "Takes no arguments. A small regression toy set."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT round(min(progression), 1) AS lo, round(max(progression), 1) AS hi, "
                    "round(avg(progression), 1) AS mean FROM lightgbm.main.diabetes()"
                ),
                description="Range and mean of the one-year progression score",
            )
        ]


_CALIFORNIA_SCHEMA = _regression_schema(
    ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
    "median_house_value",
    "Median house value for the block group, in units of $100,000.",
    _CALIFORNIA_FEATURE_DOCS,
)


@init_single_worker
@bind_fixed_schema
class CaliforniaHousingFunction(TableFunctionGenerator[NoArgs]):
    """California housing regression: 20640 districts, 8 features, median house value.

    Downloaded from scikit-learn on first use and cached under the standard
    scikit-learn data home (``~/scikit_learn_data`` or ``SCIKIT_LEARN_DATA``).
    """

    FIXED_SCHEMA: ClassVar[pa.Schema] = _CALIFORNIA_SCHEMA

    class Meta:
        name = "california_housing"
        description = "California housing prices (20640 districts, 8 features, regression)"
        categories = ["datasets", "regression", "fetched"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_CALIFORNIA_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the California-housing regression dataset: 20,640 "
                "rows, one per 1990 census block group, with 8 numeric features (median income `medinc`, "
                "house age, average rooms/bedrooms, population, occupancy, latitude, longitude), a "
                "continuous `target` (median house value in $100,000s), and a `sample_id`. A large "
                "regression dataset; downloaded from scikit-learn on first use and cached locally."
            ),
            "vgi.doc_md": (
                "**California housing dataset** — 20,640 census block groups, regression.\n\n"
                "- `sample_id` `INTEGER` — 0-based row index\n"
                "- 8 `DOUBLE` features (`medinc`, `houseage`, `averooms`, `latitude`, `longitude`, ...)\n"
                "- `target` `DOUBLE` — median house value (in $100k)\n\n"
                "Takes no arguments. Downloads on first use; a larger regression benchmark."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT round(latitude) AS lat, count(*) AS block_groups, "
                    "round(avg(median_house_value), 3) AS avg_value "
                    "FROM lightgbm.main.california_housing() GROUP BY lat ORDER BY lat"
                ),
                description="Mean median-house-value per latitude band (downloads on first use)",
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=20640, max=20640)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        bunch = skd.fetch_california_housing()
        _emit_matrix(
            bunch.data,
            {"median_house_value": [float(t) for t in bunch.target]},
            cls.FIXED_SCHEMA,
            out,
            params.output_schema,
        )


# ===========================================================================
# Synthetic generators (schema depends on arguments -> custom on_bind)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class MakeClassificationArgs:
    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=20, doc="Total number of features.")]
    n_informative: Annotated[int, Arg("n_informative", default=2, doc="Number of informative features.")]
    n_classes: Annotated[int, Arg("n_classes", default=2, doc="Number of target classes.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeClassificationFunction(TableFunctionGenerator[MakeClassificationArgs]):
    """Generate a random n-class classification problem."""

    class Meta:
        name = "make_classification"
        description = "Generate a synthetic classification dataset"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_dynamic_columns_md": result_dynamic_columns_md(
                [
                    (
                        "Always present",
                        [
                            (
                                "sample_id",
                                "INTEGER",
                                "Row index within the generated sample (0-based).",
                            ),
                            ("target", "INTEGER", "Integer class label (0-based)."),
                        ],
                    ),
                    (
                        "Generated features — the shape for `n_features := 3`",
                        [
                            ("feature_0", "DOUBLE", "First generated feature value."),
                            ("feature_1", "DOUBLE", "Second generated feature value."),
                            ("feature_2", "DOUBLE", "Third generated feature value."),
                        ],
                    ),
                ],
                note=(
                    "The result width varies with `n_features`: `sample_id` and `target` are "
                    "always present, and one `feature_<i>` `DOUBLE` column is added per "
                    "generated feature (`i` runs `0 .. n_features - 1`). The second table below "
                    "shows the concrete feature block for `n_features := 3`."
                ),
            ),
            "vgi.doc_llm": (
                "Table function that synthesises a random n-class classification problem (scikit-learn's "
                "`make_classification`). Control the shape with `n_samples`, `n_features`, `n_informative`, "
                "`n_classes`, and `random_state`; the output schema has a `sample_id`, one `feature_<i>` "
                "`DOUBLE` column per feature, and an integer `target`. Use it to generate reproducible "
                "training data of an arbitrary size/dimensionality for fit/predict, search, or benchmarks "
                "without loading a real dataset."
            ),
            "vgi.doc_md": (
                "**Synthetic classification generator** — a random n-class problem.\n\n"
                "- Args: `n_samples`, `n_features`, `n_informative`, `n_classes`, `random_state`\n"
                "- `sample_id` `INTEGER`, `target` `INTEGER` (0-based label)\n"
                "- one `feature_<i>` `DOUBLE` column per feature (width set by `n_features`)\n\n"
                "Reproducible for a fixed `random_state`; great for sizing/stress demos."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT target, count(*) AS n, round(avg(feature_0), 3) AS avg_feature_0 "
                    "FROM lightgbm.main.make_classification(n_samples := 500, n_features := 3, n_classes := 3) "
                    "GROUP BY target ORDER BY target"
                ),
                description="Class balance and mean first feature of a 500-row, 3-feature, 3-class sample",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeClassificationArgs]) -> BindResponse:
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "target", pa.int32(), "Integer class label (0-based)."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeClassificationArgs]) -> TableCardinality:
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeClassificationArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        # sklearn requires n_classes * n_clusters_per_class <= 2**n_informative.
        # Use one cluster per class and raise n_informative just enough (capped
        # at n_features) so any valid n_classes/n_features combination works.
        needed = max(1, math.ceil(math.log2(max(2, a.n_classes))))
        n_informative = min(max(a.n_informative, needed), a.n_features)
        x, y = skd.make_classification(
            n_samples=a.n_samples,
            n_features=a.n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=a.n_classes,
            n_clusters_per_class=1,
            random_state=a.random_state,
        )
        _emit_matrix(x, {"target": [int(v) for v in y]}, params.output_schema, out, params.output_schema)


@dataclass(slots=True, frozen=True)
class MakeRegressionArgs:
    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=20, doc="Total number of features.")]
    n_informative: Annotated[int, Arg("n_informative", default=10, doc="Number of informative features.")]
    noise: Annotated[float, Arg("noise", default=0.0, doc="Std-dev of gaussian noise on the output.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeRegressionFunction(TableFunctionGenerator[MakeRegressionArgs]):
    """Generate a random regression problem."""

    class Meta:
        name = "make_regression"
        description = "Generate a synthetic regression dataset"
        categories = ["datasets", "synthetic", "regression"]
        projection_pushdown = True
        tags = {
            "vgi.result_dynamic_columns_md": result_dynamic_columns_md(
                [
                    (
                        "Always present",
                        [
                            (
                                "sample_id",
                                "INTEGER",
                                "Row index within the generated sample (0-based).",
                            ),
                            ("target", "DOUBLE", "Continuous regression target."),
                        ],
                    ),
                    (
                        "Generated features — the shape for `n_features := 3`",
                        [
                            ("feature_0", "DOUBLE", "First generated feature value."),
                            ("feature_1", "DOUBLE", "Second generated feature value."),
                            ("feature_2", "DOUBLE", "Third generated feature value."),
                        ],
                    ),
                ],
                note=(
                    "The result width varies with `n_features`: `sample_id` and `target` are "
                    "always present, and one `feature_<i>` `DOUBLE` column is added per "
                    "generated feature (`i` runs `0 .. n_features - 1`). The second table below "
                    "shows the concrete feature block for `n_features := 3`."
                ),
            ),
            "vgi.doc_llm": (
                "Table function that synthesises a random linear regression problem (scikit-learn's "
                "`make_regression`). Control the shape with `n_samples`, `n_features`, `n_informative`, "
                "`noise`, and `random_state`; the output schema has a `sample_id`, one `feature_<i>` "
                "`DOUBLE` column per feature, and a continuous `DOUBLE` `target`. Use it to generate "
                "reproducible regression data of an arbitrary size with a tunable noise level for "
                "fit/predict, search, or interpretation demos."
            ),
            "vgi.doc_md": (
                "**Synthetic regression generator** — a random linear problem with optional noise.\n\n"
                "- Args: `n_samples`, `n_features`, `n_informative`, `noise`, `random_state`\n"
                "- `sample_id` `INTEGER`, `target` `DOUBLE` (continuous)\n"
                "- one `feature_<i>` `DOUBLE` column per feature (width set by `n_features`)\n\n"
                "`noise` controls the output's standard deviation; reproducible for a fixed seed."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT round(avg(target), 3) AS mean_target, round(stddev(target), 3) AS sd_target "
                    "FROM lightgbm.main.make_regression(n_samples := 500, n_features := 3, noise := 5.0)"
                ),
                description="Mean and spread of a noisy 500-row, 3-feature regression target",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeRegressionArgs]) -> BindResponse:
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "target", pa.float64(), "Continuous regression target."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeRegressionArgs]) -> TableCardinality:
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeRegressionArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        x, y = skd.make_regression(
            n_samples=a.n_samples,
            n_features=a.n_features,
            n_informative=a.n_informative,
            noise=a.noise,
            random_state=a.random_state,
        )
        _emit_matrix(x, {"target": [float(v) for v in y]}, params.output_schema, out, params.output_schema)


DATASET_FUNCTIONS: list[type] = [
    IrisFunction,
    WineFunction,
    BreastCancerFunction,
    DiabetesFunction,
    CaliforniaHousingFunction,
    MakeClassificationFunction,
    MakeRegressionFunction,
]
