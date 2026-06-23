# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.8.2",
#     "vgi-rpc[sentry]>=0.20.4",
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
    SELECT * FROM lightgbm.fit((SELECT * FROM lightgbm.iris()), model_name => 'm', target => 'target');
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_lightgbm import __version__
from vgi_lightgbm.datasets import DATASET_FUNCTIONS
from vgi_lightgbm.importance import IMPORTANCE_FUNCTIONS
from vgi_lightgbm.models import MODEL_FUNCTIONS
from vgi_lightgbm.search import SEARCH_FUNCTIONS
from vgi_lightgbm.typed_models import TYPED_FIT_FUNCTIONS

log = logging.getLogger(__name__)

DATA_VERSION = __version__
GIT_COMMIT = os.environ.get("VGI_LIGHTGBM_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by area.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *MODEL_FUNCTIONS,
    *TYPED_FIT_FUNCTIONS,
    *SEARCH_FUNCTIONS,
    *IMPORTANCE_FUNCTIONS,
]

_LIGHTGBM_CATALOG = Catalog(
    name="lightgbm",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="LightGBM train/predict model registry, datasets, and interpretation for SQL",
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
                data_version_spec=DATA_VERSION,
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
