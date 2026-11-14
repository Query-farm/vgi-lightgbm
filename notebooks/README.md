# Executed LightGBM calibration experiment

[Do more trees fix predictions beyond the training range?](01_test_calibration_extrapolation.ipynb) fits a noise-free synthetic calibration rule through SQL, keeps training and held-out subsets separate, measures error outside the training range, and verifies SHAP contribution reconstruction. The focused experiment increases boosting iterations from 20 to 80 while keeping all data fixed.

## Run from this directory

Requirements: uv, Git, Python 3.13/3.14, network for first installation, and LightGBM's platform OpenMP runtime (libomp on macOS or libgomp on Linux). The verified workflow ran on macOS arm64. No service credentials, external dataset or global Jupyter kernel is needed.

```sh
cd notebooks
uv sync --frozen
uv run --frozen python execute.py --timeout 600
```

Copy this whole directory to run without the source checkout. Its separate lock pins published Haybarn **2.0.0rc11** and visualization dependencies. A temporary kernel uses the runner's interpreter with notebook-directory cwd. Successful outputs replace the notebook atomically; failed runs preserve the saved default.

For interactive use:

```sh
uv run --frozen python -m ipykernel install --user --name vgi-lightgbm-notebooks --display-name 'Python (LightGBM notebooks)'
```

Use this kernel with cwd here; edit the visible parameter, restart and run all. Python setup, plotting and cleanup are folded. Analytical SQL, connection and useful outputs remain visible.

## One focused experiment

`N_ESTIMATORS=20` accepts integers 1–200. All other training settings and data stay fixed.

```sh
uv run --frozen python execute.py --timeout 600 --parameters '{"N_ESTIMATORS":80}' --output-dir artifacts/trees-80
```

Reset 20 for defaults. Overrides require a separate output directory. Successful copies receive helpers, launcher, assets, README, runner and environment. Artifacts and virtual environments are ignored. A fresh temporary model registry is isolated per run and deleted at cleanup; no model from an earlier notebook run is reused.

## Literal portable launch

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'lightgbm' (TYPE vgi, LOCATION 'python3 worker_launcher.py');
```

The launcher fetches immutable **remote Git revision 48e4b113f4f80615aa7d9a6d54434035a2425961**, verifies checkout identity, and runs its unchanged `lightgbm_worker.py` entry point. The project wheel packages its implementation package but omits this top-level entry module, so the launcher retains the remote tree instead of relying on a missing installed command. It discovers no local worker source or `.venv` and builds no engine.

The worker runtime pins Python 3.13, vgi-python 0.51.1, vgi-rpc 0.53.1, PyArrow 26.0.1, LightGBM 4.6.0, scikit-learn 1.9.0, NumPy 2.5.0 and SciPy 1.18.0 through uvx. SDK 0.51.1 supports the published client's protocol 3.3; root lockfiles are preserved. The notebook environment is separate from that worker runtime.

A bounded lock and atomic installation protect `$XDG_CACHE_HOME/vgi-lightgbm-notebooks` (default `~/.cache/vgi-lightgbm-notebooks`); each reuse verifies all worker Python source hashes. A timed-out lock identifies the installer to inspect. First use needs network; later uses reuse verified source and uv caches. To upgrade, change revision/package pins deliberately and rerun default/experiment from an empty worker cache.

## Data, units and limits

Visible SQL authors 241 fixed grid points at 0.05 V intervals from 0 through 12 V, with reference response **2 mm + 3 mm/V × voltage**. Within 0–8 V, 129 rows train and 32 fixed rows are held out; 80 rows above 8 V test extrapolation. There is no real sensor, measurement noise, random sample, data download or physical validity claim. Fixed seed, row order and single-thread execution make the small experiment reproducible.

Fit receives only training rows and the voltage feature. ID and target columns are excluded from features. CREATE TABLE materializes the fit before capture, avoiding duplicate stateful calls. Prediction and explain run through the real worker; SQL joins references afterward, computes MAE/RMSE and verifies contribution sums. Python never trains a model or computes model predictions.

Error units are millimetres. Training error is not held-out performance. For this regression objective, base plus feature contributions reconstruct prediction; contributions are not causes or uncertainty estimates. The beyond-range grid is a separate condition, not evidence about arbitrary new data. Increasing trees does not make numeric tree splits extrapolate a linear physical law.

## Validation

Validated with published Haybarn 2.0.0rc11 on 2026-11-14: default 5.3 s; standalone cold-source-cache default 30.3 s; separate 80-tree experiment 12.2 s. Each run executed 14 code cells, eight SQL cells, two charts and ten invariants. All four charts were inspected, three runner guards preserved the successful default, and two concurrent cold-cache attachments completed with one verified atomic source installation.

Increasing iterations 20→80 reduced in-range holdout MAE from 0.759 to 0.179 mm and beyond-range MAE from 8.393 to 6.440 mm. The beyond-range response stayed flat: its plateau moved from 23.68 to 25.64 mm. The reference and split were unchanged. Prediction-contribution reconciliation passed in both runs.

Source checks passed: 63 unit tests (two existing sklearn feature-name warnings), all five SQLLogic test files through the repository’s real Haybarn/stdio runner (98 statement/query directives), Ruff, mypy and formatting. Both configured metadata gates passed: 100/100 L1 static and 100/100 L2 executed, zero errors/warnings and eight informational findings (title wording, dataset-ID foreign-key suggestions and plural naming). These are not a zero-finding result.

The initial locked source environment used an older Haybarn and SDK 0.50.0; its five SQL files failed protocol routing. An environment-only override to published Haybarn 2.0.0rc11 and SDK 0.51.1 reran all units/SQL successfully. Root source and uv.lock remain unchanged; the notebook launcher explicitly pins the compatible runtime.
