"""Run the unchanged worker entry point from one verified remote Git revision."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REVISION = "48e4b113f4f80615aa7d9a6d54434035a2425961"
REPOSITORY = "https://github.com/Query-farm/vgi-lightgbm.git"
RUNTIME = [
    "uvx",
    "--python",
    "3.13",
    "--from",
    "vgi-python[http,oauth]==0.51.1",
    "--with",
    "vgi-rpc[sentry]==0.53.1",
    "--with",
    "pyarrow==26.0.1",
    "--with",
    "lightgbm==4.6.0",
    "--with",
    "scikit-learn==1.9.0",
    "--with",
    "numpy==2.5.0",
    "--with",
    "scipy==1.18.0",
    "python",
]


def source_hashes(source):
    paths = [source / "lightgbm_worker.py", *sorted((source / "vgi_lightgbm").rglob("*.py"))]
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def ensure_source():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vgi-lightgbm-notebooks"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / REVISION
    source = cache / "source"
    marker = cache / "verified.json"

    def ready():
        if not marker.is_file() or not (source / "lightgbm_worker.py").is_file():
            return False
        info = json.loads(marker.read_text())
        return info["revision"] == REVISION and info["source_sha256"] == source_hashes(source)

    lock = root / f"{REVISION}.lock"
    deadline = time.monotonic() + 180
    while not ready():
        try:
            lock.mkdir()
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Source cache lock timed out: {lock}; inspect the installer before removing it"
                ) from None
            time.sleep(0.2)
            continue
        try:
            if ready():
                continue
            if cache.exists():
                raise RuntimeError(f"Source cache failed verification: {cache}; remove that cache and retry")
            if shutil.which("git") is None or shutil.which("uvx") is None:
                raise RuntimeError("Install Git and uv before running this notebook")
            with tempfile.TemporaryDirectory(prefix="install-", dir=root) as temporary:
                install = Path(temporary) / "ready"
                install.mkdir()
                checkout = install / "source"
                subprocess.run(
                    ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(checkout)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["git", "-C", str(checkout), "checkout", "--detach", REVISION],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                actual = subprocess.check_output(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
                ).strip()
                if actual != REVISION:
                    raise RuntimeError("Remote worker revision mismatch")
                (install / "verified.json").write_text(
                    json.dumps({"revision": REVISION, "source_sha256": source_hashes(checkout)})
                )
                install.rename(cache)
        finally:
            lock.rmdir()
    return source


def main():
    entry = ensure_source() / "lightgbm_worker.py"
    os.execvp(RUNTIME[0], [*RUNTIME, str(entry)])


if __name__ == "__main__":
    main()
