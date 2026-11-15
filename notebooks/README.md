# LightGBM examples

Two independent notebooks audit local model behavior with invented inputs:

- [Do more trees fix predictions beyond the training range?](01_test_calibration_extrapolation.ipynb) preserves the calibration/extrapolation story and adds a complete disposable registry lifecycle: inspect the fitted model, reconcile its metadata, then delete it and verify repeat removal.
- [Do gain, shuffle loss and average response tell the same story?](02_gain_shuffling_and_average_response.ipynb) generates a fixed regression sample, compares gain and permutation loss on separate scales, retains a constant-zero control and reconstructs a partial-dependence point with actual model predictions.

## Run locally or in Colab

Use a notebook's Colab badge and run all. Its bootstrap installs the base shared package and retrieves two immutable hash-verified companions. A local copy of this directory requires [uv](https://docs.astral.sh/uv/), Git and LightGBM's platform OpenMP runtime (`libomp` on macOS, `libgomp` on Linux):

```bash
uv sync --locked
uv run --locked vgi-notebooks 01_test_calibration_extrapolation.ipynb
uv run --locked vgi-notebooks 02_gain_shuffling_and_average_response.ipynb
```

Local publication uses the shared `[runner]` extra. Colab installs only the base package, preserving preloaded kernel/Jupyter libraries. An optional interactive kernel:

```bash
uv run --locked python -m ipykernel install --user --name vgi-lightgbm-notebooks --display-name 'Python (VGI LightGBM examples)'
```

Both books keep the published release visible:

```sql
ATTACH 'lightgbm' (TYPE vgi, LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-lightgbm@v0.1.0 python lightgbm_release.py');
```

The release wheel includes `vgi_lightgbm` but omits `lightgbm_worker.py` and a console entrypoint. The small worker-specific companion verifies that the installed Git package is exactly release commit **51375ea0956f0dcd6441754213d7726dc16f12db**, downloads only that unchanged top-level file from the fixed public commit, checks SHA256 **d57ba862014655608ddaef676c67a68fe8eed41c88770fcb8860ebdabf1e9fad**, and runs it against the installed implementation package. It accepts **no arguments**, arbitrary URLs, revisions or commands. No source checkout discovery, worker build or transitive dependency override is used.

The entry cache defaults to `~/.cache/vgi-lightgbm-release`; `VGI_LIGHTGBM_RELEASE_CACHE` can select a dedicated cache directory. A bounded POSIX file lock and atomic write protect simultaneous starts, and every reuse rechecks file bytes. The launcher reports failures on stderr while worker stdout remains Arrow protocol. First use needs public packages/the fixed worker file and the published Haybarn/VGI extension. Inference and training are local; no account, API key or downloaded model is needed. Actual resolved package versions are [recorded](coverage/resolved-worker-packages.json); release dependency ranges can resolve differently later and upgrades need validation.

Each notebook creates a fresh temporary local model registry, temporarily sets `LIGHTGBM_MODELS_DIR`, and restores the previous value after removing its own directory. SQL connections separately use a new empty temporary secret directory with persistent secrets disabled. No existing model registry or saved SQL credentials is opened. Named-model writes/deletion affect only notebook-created models.

## Focused experiments

```bash
uv run --locked vgi-notebooks 01_test_calibration_extrapolation.ipynb \
  --parameters '{"N_ESTIMATORS":80}' --output-dir artifacts/more-trees
uv run --locked vgi-notebooks 02_gain_shuffling_and_average_response.ipynb \
  --parameters '{"EVALUATION_SPLIT":"training"}' --output-dir artifacts/training-background
```

The first changes 20 to 80 boosting iterations while preserving all 241 voltage/reference rows and their 129 training, 32 in-range held-out and 80 outside-range roles. The second switches 150 held-out evaluation rows to 450 training rows while preserving all 600 generated rows, training inputs, estimator settings, gain values and full-population model predictions. The partial-dependence grid can change with its background population. Packed model bytes include creation metadata, so raw model-BLOB hash identity is not claimed across independent runs.

The runner requires separate experiment destinations, rejects unknown parameters/unsafe paths, and preserves the saved default after failure. SVG charts and content-addressed GitHub images accompany the defaults; ignored artifacts contain logs, review PNGs and experiments.

## Meaning and limits

The calibration rule is authored, noise-free **2 mm + 3 mm/V × voltage**. It is not a real sensor or physical law. Held-out grid error and outside-range error have distinct denominators. Regression SHAP contributions reconstruct model predictions in target units; they are not causes or uncertainty bounds.

The second sample comes from the worker's seeded regression generator: 600 rows, three generated features, two informative terms and noise standard deviation 5. The generator does not return its coefficients, so the notebook does not claim to identify the true informative feature names. SQL adds a constant-zero feature. Gain uses objective-dependent units; permutation uses negative MAE, so mean score loss is MAE increase in target units. Five-shuffle standard deviation is not a confidence interval, and small/zero/negative values remain in the table. Partial dependence is an averaged model query with one feature replaced, not an intervention or observed response. Changing evaluation population is a diagnostic comparison, not tuning or an independent accuracy estimate for training rows.

## Coverage and validation

The actual complete release catalog contains **22 logical named surfaces**: 22 function rows and six same-named physical table aliases. There are no views or COPY handlers. Registry-writing fits/search and `drop_model` remain in the denominator. These two stories meaningfully cover **11/22**, meeting the required 11. The 11 uncovered names are `iris`, `wine`, `breast_cancer`, `diabetes`, `california_housing`, `make_classification`, `fit_lgbm_classifier`, `cross_val_predict`, `cross_val_score`, `grid_search` and `randomized_search`. A startup smoke on iris is not counted as narrative coverage.

[coverage.json](coverage.json) binds full native catalogs/arguments, static/dynamic schema metadata, visible SQL, interpreted results and successful execution receipts. Release v0.1.0 peels to the exact current-main source commit **51375ea0956f0dcd6441754213d7726dc16f12db**; both catalogs and receipts use that full revision, so **11/22 is exact released and exact current runtime coverage** for the audited source. Proper heading title/subtitle come from actual catalog metadata.

The books contain ten and twelve SQL invariants respectively, plus six registry assertions in the first. Defaults and separate experiments execute the actual release. Two simultaneous attachments passed from an empty entry-file cache. Detached notebook runs and a fresh Python 3.12 kernel start with empty companion/entry caches; the kernel preloads eight plotting/Jupyter libraries and checks unchanged versions plus `pip check`. These are macOS ARM/Colab-compatible setup checks, not manual hosted-Colab UI, Linux or Windows execution. Upstream native OpenMP was already installed on the validation host. All default/experiment figures are reviewed and published SVG bytes verified.

Worker code and root dependencies remain unchanged; [green source CI](https://github.com/Query-farm/vgi-lightgbm/actions/runs/43615456249) is reused. The separate existing [deployment workflow failure](https://github.com/Query-farm/vgi-lightgbm/actions/runs/43615457355) is retained as baseline evidence and is neither retried nor addressed by notebook work. No worker rebuild, repeated full source suite, deployment or container publication is requested. [STANDARD.md](STANDARD.md) links the shared standard.
