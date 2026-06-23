"""Model registry: persist fitted LightGBM boosters behind a swappable storage backend.

Ships a local-disk store (``LIGHTGBM_MODELS_DIR``, default ``./models``). The
``ModelStore`` interface is the seam where an S3/R2 backend drops in later
without touching ``models.py``.

**Serialization is LightGBM's native text format, not pickle.** Each fitted
estimator is reduced to its ``Booster`` text (``booster_.model_to_string()``)
plus a small JSON metadata record; loading reconstructs a ``lightgbm.Booster``
from the text. This avoids arbitrary-code-execution risk (unlike pickle) and is
portable across LightGBM patch versions. The sklearn wrapper's class labels and
task are carried in metadata, so prediction (label / proba) is reproduced from
the raw booster without unpickling a Python object.

Each model on disk is two artifacts:
* ``<name>.lgb``  -- the LightGBM booster text
* ``<name>.json`` -- ``ModelMetadata`` (estimator type, ordered feature names,
  target, classes, categorical mask + category orderings, hyperparameters, train
  score, library versions, timestamp)

The same (booster-text, metadata) pair is also packed into a self-describing
``model`` BLOB by ``pack_model`` so ``predict`` can take a model inline without a
registry round-trip (see ``models.py``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_BLOB_MAGIC = b"VGILGB01"


class ModelNameError(ValueError):
    """Raised for model names that are empty or unsafe as a filename."""


class ModelNotFoundError(KeyError):
    """Raised when a requested model is not in the registry."""


class ModelBlobError(ValueError):
    """Raised when a model BLOB cannot be decoded."""


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name) or "/" in name or ".." in name:
        raise ModelNameError(
            f"invalid model name {name!r}: use letters, digits, '_', '-', '.' and do not start with a separator"
        )
    return name


@dataclass(kw_only=True)
class ModelMetadata:
    """Everything needed to score new data and describe a stored model."""

    name: str
    estimator: str
    task: str  # "classification" | "regression"
    target: str
    feature_names: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    classes: list[Any] | None = None
    categorical: list[bool] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0
    train_score: float | None = None
    lightgbm_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelMetadata:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


def booster_to_text(estimator: Any) -> str:
    """Reduce a fitted estimator (or Booster) to its LightGBM model text."""
    booster = estimator.booster_ if hasattr(estimator, "booster_") else estimator
    return booster.model_to_string()


def booster_from_text(text: str) -> lgb.Booster:
    """Reconstruct a ``lightgbm.Booster`` from its model text."""
    return lgb.Booster(model_str=text)


# ===========================================================================
# Model BLOB: self-describing (booster text + metadata) for inline predict.
# ===========================================================================


def pack_model(estimator: Any, meta: ModelMetadata) -> bytes:
    """Pack a fitted model + metadata into a portable, self-describing BLOB.

    Layout: magic + uint32 metadata-length + UTF-8 JSON metadata + booster text.
    """
    text = booster_to_text(estimator).encode("utf-8")
    meta_bytes = json.dumps(meta.to_dict(), default=str).encode("utf-8")
    header = _BLOB_MAGIC + len(meta_bytes).to_bytes(4, "big")
    return header + meta_bytes + text


def unpack_meta(blob: bytes) -> ModelMetadata:
    """Read just the metadata from a model BLOB (used at bind time)."""
    if blob[:8] != _BLOB_MAGIC:
        raise ModelBlobError("not a LightGBM model BLOB (bad magic header)")
    mlen = int.from_bytes(blob[8:12], "big")
    return ModelMetadata.from_dict(json.loads(blob[12 : 12 + mlen].decode("utf-8")))


def unpack_model(blob: bytes) -> tuple[lgb.Booster, ModelMetadata]:
    """Read the booster + metadata from a model BLOB (used at process time)."""
    meta = unpack_meta(blob)
    mlen = int.from_bytes(blob[8:12], "big")
    text = blob[12 + mlen :].decode("utf-8")
    return booster_from_text(text), meta


# ===========================================================================
# Stores
# ===========================================================================


class ModelStore:
    """Abstract model store. Implementations persist (booster-text, metadata) by name."""

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        raise NotImplementedError

    def load(self, name: str) -> tuple[lgb.Booster, ModelMetadata]:
        raise NotImplementedError

    def load_meta(self, name: str) -> ModelMetadata:
        raise NotImplementedError

    def list(self) -> list[ModelMetadata]:
        raise NotImplementedError

    def delete(self, name: str) -> bool:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError


class LocalDiskStore(ModelStore):
    """Stores models as ``<root>/<name>.lgb`` + ``<root>/<name>.json``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        validate_name(name)
        return self.root / f"{name}.lgb", self.root / f"{name}.json"

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        model_path, meta_path = self._paths(meta.name)
        model_path.write_text(booster_to_text(estimator))
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2, default=str))

    def load(self, name: str) -> tuple[lgb.Booster, ModelMetadata]:
        model_path, _ = self._paths(name)
        if not model_path.exists():
            raise ModelNotFoundError(name)
        return booster_from_text(model_path.read_text()), self.load_meta(name)

    def load_meta(self, name: str) -> ModelMetadata:
        _, meta_path = self._paths(name)
        if not meta_path.exists():
            raise ModelNotFoundError(name)
        return ModelMetadata.from_dict(json.loads(meta_path.read_text()))

    def list(self) -> list[ModelMetadata]:
        if not self.root.exists():
            return []
        out: list[ModelMetadata] = []
        for meta_path in sorted(self.root.glob("*.json")):
            try:
                out.append(ModelMetadata.from_dict(json.loads(meta_path.read_text())))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def delete(self, name: str) -> bool:
        model_path, meta_path = self._paths(name)
        existed = model_path.exists() or meta_path.exists()
        model_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed

    def exists(self, name: str) -> bool:
        model_path, _ = self._paths(name)
        return model_path.exists()


_store: ModelStore | None = None


def get_store() -> ModelStore:
    """Return the process-wide model store, configured from the environment.

    ``LIGHTGBM_MODELS_DIR`` selects the local-disk root (default ``./models``).
    A future S3/R2 backend would be selected here behind the same interface.
    """
    global _store
    if _store is None:
        root = os.environ.get("LIGHTGBM_MODELS_DIR", "models")
        _store = LocalDiskStore(root)
    return _store


def set_store(store: ModelStore | None) -> None:
    """Override the process-wide store (used by tests)."""
    global _store
    _store = store


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
