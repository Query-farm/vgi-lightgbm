"""Run the authoritative ``test/sql/*.test`` files in-process via haybarn.

This is the CI-portable form of the SQL integration suite: it loads the ``vgi``
community extension into Query Farm's ``haybarn`` DuckDB build, attaches the
worker over stdio, and replays the same ``.test`` files the local ``unittest``
runner uses. Opt-in via ``VGI_SQL_HAYBARN=1`` (set in CI) so the fast offline
``make pytest`` does not spawn workers or hit the network.
"""

from __future__ import annotations

import glob
import os
import sys

import pytest

if not os.environ.get("VGI_SQL_HAYBARN"):
    pytest.skip("set VGI_SQL_HAYBARN=1 to run the haybarn SQL suite", allow_module_level=True)

haybarn = pytest.importorskip("haybarn")

from tests.sqllogic import run_test_file  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "test", "sql", "*.test")))


@pytest.fixture(scope="session", autouse=True)
def _worker_env(tmp_path_factory) -> None:
    # Point the worker at the same interpreter running pytest (deps already
    # installed) and an isolated registry so the suite is hermetic.
    os.environ["VGI_LIGHTGBM_WORKER"] = f"{sys.executable} lightgbm_worker.py"
    os.environ["LIGHTGBM_MODELS_DIR"] = str(tmp_path_factory.mktemp("models"))


@pytest.mark.parametrize("test_file", _TEST_FILES, ids=lambda p: os.path.basename(p))
def test_sql_file(test_file: str) -> None:
    con = haybarn.connect()
    con.execute("INSTALL vgi FROM community")
    con.execute("LOAD vgi")
    try:
        run_test_file(test_file, con)
    finally:
        con.close()


def _published_example_sqls() -> list[tuple[str, str]]:
    """Every published example SQL: per-function lists, schema list, executable examples.

    The model-reading functions are exempt from vgi-lint's live example check
    (VGI901/902 -- no stored model exists in a fresh session), so nothing else
    would notice one of their examples going stale (e.g. a renamed column).
    """
    import json

    import lightgbm_worker as w

    out: list[tuple[str, str]] = []
    for fn in w._FUNCTIONS:
        for ex in getattr(fn.Meta, "examples", []) or []:
            out.append((fn.Meta.name, ex.sql))
    for ex in json.loads(w._SCHEMA_TAGS["vgi.example_queries"]):
        out.append(("schema", ex["sql"]))
    for ex in json.loads(w._CATALOG_EXECUTABLE_EXAMPLES):
        stmts = ex["sql"] if isinstance(ex["sql"], list) else [ex["sql"]]
        out.extend(("executable", s) for s in stmts)
    # california_housing downloads its data on first use; keep this test offline.
    return [(name, sql) for name, sql in out if "california_housing" not in sql]


def test_published_examples_execute() -> None:
    """Each published example binds and runs once the models they name are stored."""
    con = haybarn.connect()
    con.execute("INSTALL vgi FROM community")
    con.execute("LOAD vgi")
    try:
        con.execute(f"ATTACH 'lightgbm' AS lightgbm (TYPE vgi, LOCATION '{os.environ['VGI_LIGHTGBM_WORKER']}')")
        # The stored models the illustrative examples refer to by name.
        con.execute(
            "SELECT model_name FROM lightgbm.main.fit((SELECT * EXCLUDE (target_name) FROM lightgbm.main.iris()), "
            "model_name := 'iris_clf', estimator := 'lgbm_classifier', target := 'target', id := 'sample_id')"
        ).fetchall()
        con.execute(
            "SELECT model_name FROM lightgbm.main.fit((SELECT * FROM lightgbm.main.diabetes()), "
            "model_name := 'diab_reg', estimator := 'lgbm_regressor', target := 'progression', id := 'sample_id')"
        ).fetchall()
        failures = []
        for name, sql in _published_example_sqls():
            try:
                con.execute(sql).fetchall()
            except Exception as exc:  # collect every failure
                failures.append(f"{name}: {exc}\n  {sql}")
        assert not failures, "\n".join(failures)
    finally:
        con.close()
