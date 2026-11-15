"""Load the fixed entry module omitted from the v0.1.0 LightGBM wheel."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REVISION = "51375ea0956f0dcd6441754213d7726dc16f12db"
ENTRY_SHA256 = "d57ba862014655608ddaef676c67a68fe8eed41c88770fcb8860ebdabf1e9fad"
ENTRY_URL = f"https://raw.githubusercontent.com/Query-farm/vgi-lightgbm/{REVISION}/lightgbm_worker.py"


def verified_entry() -> Path:
    """Cache only the immutable released entry module with an atomic write."""
    distribution = importlib.metadata.distribution("vgi-lightgbm")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if distribution.version != "0.1.0" or origin.get("vcs_info", {}).get("commit_id") != REVISION:
        raise RuntimeError("Launch with the documented vgi-lightgbm v0.1.0 Git release command")
    root = Path(os.environ.get("VGI_LIGHTGBM_RELEASE_CACHE", Path.home() / ".cache/vgi-lightgbm-release"))
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{REVISION}-lightgbm_worker.py"
    with (root / "install.lock").open("a") as lock:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the LightGBM release cache lock") from None
                time.sleep(0.1)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != ENTRY_SHA256:
                raise RuntimeError("Cached LightGBM entry module failed its fixed SHA256 check")
            return target
        request = urllib.request.Request(ENTRY_URL, headers={"User-Agent": "vgi-lightgbm-notebook/0.1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != ENTRY_SHA256:
            raise RuntimeError("Published LightGBM entry module failed its fixed SHA256 check")
        with tempfile.NamedTemporaryFile(dir=root, prefix="entry-", delete=False) as temporary:
            staging = Path(temporary.name)
            temporary.write(payload)
        try:
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
    return target


def main() -> None:
    """Run the unchanged released worker; no arbitrary download or command API."""
    if len(sys.argv) != 1:
        raise ValueError("The LightGBM release companion accepts no arguments")
    entry = verified_entry()
    os.environ["VGI_LIGHTGBM_GIT_COMMIT"] = REVISION
    sys.argv = [str(entry)]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
