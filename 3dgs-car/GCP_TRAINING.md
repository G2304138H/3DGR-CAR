# Monocular Gaussian Center Predictor

This implementation adds the learned initialization stage described by
3DGR-CAR while retaining the existing FDK/BP baselines. Each stored projection
is one monocular GCP training sample. During reconstruction, only the first
selected view initializes the Gaussians; all selected views are used by the
subsequent projection optimization.

## Paired data

The projection NPZ must supply `images`, `theta_deg`, and `phi_deg`. Normally it
also follows the Stage-2 contract documented in `STAGE2_NPZ_RECONSTRUCTION.md`
and supplies `sid` and `imager_pixel_spacing`. Older archives may obtain either
missing value from `fallback_sid_m` and
`fallback_detector_pixel_spacing_mm` in the training configuration. A value
stored in an archive always takes precedence over its fallback. Optional fields
include `projection_center_offset`, `vessel_type`, and `case_id`.

The matched ground-truth NPZ supplies a binary 3D volume and physical spacing.
For the ImageCAS files used here:

- `vol` is stored in `[x,y,z]` order;
- `spacing` contains XYZ voxel spacing in millimetres.

Ground-truth files may be organized as `<gt-root>/<vessel_type>/<case_id>.npz`,
for example `lca/1.npz`. The preprocessing code samples each full-size mask on
the same centered physical cube used by reconstruction, extracts a normalized
ZYX foreground point cloud, and ray-casts a first-hit depth target for every
view. Derived targets can be cached with `--cache-dir`.

The 3D mask must cover the same vessel territory as its projections. In
particular, an LCA-only projection must not be paired with a whole-coronary mask
that also contains an invisible RCA. If a source volume contains both trees,
split or rasterize a vessel-specific mask before training. The ImageCAS-style
NPZ contract also assumes zero physical origin and positive XYZ directions,
because those archives contain spacing but no origin/direction metadata.

Depth is normalized between each ray's entry and exit points in the centered
cube. Empty rays use depth `1`; a separate `depth_mask` distinguishes empty
rays from foreground at the exit face.

No stored centerline or skeleton target is required. The GCP point cloud and
depth supervision are derived from the dense 3D mask. Stage-2 derives its 2D
centerline weighting directly from each target projection.

## Train

The split JSON may contain top-level `train`, `validation`/`val`, and `test`
lists or those lists nested under `splits`. Case entries may be strings or
records with a case-name/id field. Feature-file paths are also accepted. For
example, `.../lca/1/prefix_02.npz` is matched to `lca_0001.npz`, and
`.../rca_0508.npz` is matched to `rca_0508.npz`.

Two ready-to-run configurations are provided:

- `configs/gcp_imagecas_lca.json` uses the LCA projection directory and checks
  for `0.65` mm detector pixels, falling back to `0.65` mm when that field is
  absent.
- `configs/gcp_imagecas_rca.json` uses the RCA projection directory and checks
  for `0.55` mm detector pixels, falling back to `0.55` mm and `0.9` m SID
  when those fields are absent.

The check does not override calibration. Ray geometry uses calibration stored
in each individual projection NPZ when available and the configured fallback
otherwise. The resulting default reconstruction extent is approximately
`0.1387` m for LCA and `0.1173` m for RCA with a 256-pixel detector, 0.75 m
source-to-isocentre distance, and 0.9 m SID.

Train separate LCA and RCA predictors with these independently generated split
files. Combining them into one model could put the LCA and RCA of the same
physical patient into different train/validation partitions.

## Python environment on the training server

Load the cluster's Python module before creating or activating the environment.
The system `/usr/bin/python3` may not include `ensurepip`, whereas the
module-provided interpreter normally does:

```bash
module load python3
which python3
python3 --version
python3 -m venv --clear /export/home2/reny0012/vir_env/3dgr_car_gcp
source /export/home2/reny0012/vir_env/3dgr_car_gcp/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If an existing environment is active, run `deactivate` before loading the
module and creating the new environment. In later login sessions, activate the
environment in the same way as `vesseltree`:

```bash
module load python3
source /export/home2/reny0012/vir_env/3dgr_car_gcp/bin/activate
```

If `venv` still reports that `ensurepip` is unavailable after loading the
module, bootstrap `virtualenv` into a private directory without sudo:

```bash
python3 -m pip --version
python3 -m pip install --upgrade \
  --target /export/home2/reny0012/vir_env/virtualenv_bootstrap \
  virtualenv

PYTHONPATH=/export/home2/reny0012/vir_env/virtualenv_bootstrap \
  python3 -m virtualenv --clear --python python3 \
  /export/home2/reny0012/vir_env/3dgr_car_gcp

/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python -m pip install --upgrade pip setuptools wheel
```

The fallback's `--clear` flag removes the incomplete environment left by a
failed `venv` command. Omit it when creating the environment for the first
time.

Install the CUDA build of PyTorch supported by the server driver. For the
currently supported CUDA 12.6 wheel:

```bash
/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cu126
/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python -m pip install -r requirements-gcp.txt
```

Verify the interpreter and GPU before starting a long run:

```bash
/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python -c \
  "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

GCP training itself does not require ASTRA. To use the trained checkpoint in
the complete Stage-2 optimizer later, install the remaining pip dependencies:

```bash
/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python -m pip install -r requirements-stage2.txt
```

```bash
python train_gcp.py \
  --projection-dir /path/to/projection_npzs \
  --ground-truth-dir /path/to/volume_npzs \
  --split-json /path/to/case_split.json \
  --train-split train \
  --validation-split val \
  --output-dir ./outputs/gcp \
  --cache-dir ./outputs/gcp_target_cache \
  --image-size 128 \
  --volume-size 128 \
  --downsample-factor 2 \
  --source-origin-distance-m 0.75 \
  --max-points 10000 \
  --epochs 100 \
  --batch-size 1 \
  --amp
```

For the supplied server paths, run one vessel at a time from `3dgs-car`:

```bash
cd /path/to/3DGR-CAR/3dgs-car

/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python train_gcp.py \
  --config configs/gcp_imagecas_lca.json

/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python train_gcp.py \
  --config configs/gcp_imagecas_rca.json
```

CLI values override the JSON settings. For example, to select a GPU and resume
an interrupted LCA run:

```bash
/export/home2/reny0012/vir_env/3dgr_car_gcp/bin/python train_gcp.py \
  --config configs/gcp_imagecas_lca.json \
  --device cuda:1 \
  --resume /export/home2/reny0012/result/3dgr_car_gcp/lca/last_gcp.pt
```

Outputs include `training_config.json`, `history.jsonl`, `last_gcp.pt`, and the
validation-selected `best_gcp.pt`. Checkpoints contain the complete model
configuration and coordinate parameterization needed by Stage-2 inference.

## Reconstruct with learned initialization

```bash
python train_stage2_npz.py \
  --input /path/to/lca_0001.npz \
  --output-dir ./outputs/lca_0001_gcp \
  --view-indices 0 1 \
  --init-method gcp \
  --gcp-checkpoint ./outputs/gcp/best_gcp.pt \
  --iterations 8000
```

In this example, view `0` is the sole GCP input. Views `0` and `1` both enter
the Gaussian projection loss. The GCP path defaults to the paper-inspired loss

```text
0.5 * full_image_mse + 0.5 * centerline_masked_mse
```

Pass `--projection-loss-alpha` to change the first weight. FDK/BP runs retain
their previous full-image-MSE behavior unless that option is supplied.
Use the same `--source-origin-distance-m` value for GCP training and Stage-2
reconstruction; it defaults to `0.75` metres in both commands.

For split evaluation, `evaluate_stage2_npz.py` forwards the GCP arguments:

```bash
python evaluate_stage2_npz.py \
  --input-dir /path/to/projection_npzs \
  --ground-truth-dir /path/to/volume_npzs \
  --split-json /path/to/case_split.json \
  --split test \
  --output-dir ./outputs/gcp_test \
  --view-indices 0 1 \
  --init-method gcp \
  --gcp-checkpoint ./outputs/gcp/best_gcp.pt \
  --iterations 8000
```

## Config-driven validation and test evaluation

`evaluate_gcp.py` is the high-level evaluation entry point. It follows the
parametric evaluator's JSON conventions. The nested `model` object holds the
experiment directory, pretrained weights, checkpoint choice, and predictor
architecture; the remaining fields select the mode, split, view counts, data,
optimization, metrics, and output directory. When `training_config.json` is
present in the GCP experiment, missing data and calibration paths can still be
inherited automatically; any non-null evaluation value overrides them.

Start from one of these templates:

- `configs/eval_gcp_paper_metric_template.json`
- `configs/eval_gcp_visualisation_template.json`

Then run:

```console
python -m evaluate_gcp --config configs/eval_gcp_paper_metric_template.json
```

Use `--dry-run` first to resolve all paths, checkpoint choices, view prefixes,
and Stage-2 arguments without starting CUDA reconstruction. The resolved files
are saved as `resolved_config.json` and `evaluation_plan.json`.

For `evaluation_mode: "paper_metric"`, every selected validation/test case is
initialized by the GCP and optimized independently. Results are grouped by
view count under `metrics/by_view_count/k<N>`, with parametric-style combined
files at the root:

- `performance_per_case.json`
- `performance_summary.json`
- `metrics/paper_metric_per_case.json` and `.csv`
- `metrics/paper_metric_summary.json`
- `metrics/metrics_matrix.npz`
- `metrics/voxel_masks/manifest.json`

For `evaluation_mode: "visualisation"`, the same metrics are computed and the
full per-case Stage-2 artifact bundle is retained under
`visualization/k<N>/cases/<case>`. This includes Gaussian checkpoints, the
reconstructed volume and GIF, input/novel reprojections, and run metadata.
`max_visualizations` limits full artifact generation without reducing the set
of cases that receive optimization and metrics.

`eval_num_views` may be one integer or a list such as `[1, 2]`.
`eval_view_indices` defines the fixed order, and each view-count run uses its
prefix: with `[3, 5]`, the one-view run uses `[3]` and the two-view run uses
`[3, 5]`. Only the first selected view enters the monocular GCP; all selected
views constrain the subsequent Gaussian optimization.

### Run the LCA and RCA validation-plus-test evaluations with Python

The repository includes one self-contained paper-metric configuration for each
separately trained predictor:

- `configs/eval_gcp_paper_metric_lca_val_test.json`
- `configs/eval_gcp_paper_metric_rca_val_test.json`

Each file selects `best_gcp.pt`, evaluates every validation case followed by
every test case through the combined `val_test` split, and stores the predictor
architecture, paths, detector calibration, Gaussian optimizer parameters, and
paper-metric settings. The architecture in the file is checked against the
architecture saved in the checkpoint before inference. Stored NPZ calibration
takes precedence over the configured fallback values.

From the `3dgs-car` directory, run LCA with:

```console
python -m evaluate_gcp --config configs/eval_gcp_paper_metric_lca_val_test.json
```

Run RCA with:

```console
python -m evaluate_gcp --config configs/eval_gcp_paper_metric_rca_val_test.json
```

Add `--dry-run` to either command to resolve the paths and write the evaluation
plan without running any Gaussian optimization.

### Visualize one selected LCA or RCA case

`visualize_gcp_case.py` provides the single-case equivalent of the parametric
model's visualization evaluation. It selects one artery-specific configuration
and one numeric case ID, then runs the complete GCP-to-Gaussian pipeline.

For example, visualize RCA case 508 with:

```console
python -m visualize_gcp_case --artery rca --case-number 508
```

Visualize LCA case 17 with:

```console
python -m visualize_gcp_case --artery lca --case-number 17
```

The settings are stored in:

- `configs/eval_gcp_visualisation_rca_case.json`
- `configs/eval_gcp_visualisation_lca_case.json`

The configurations use the combined `val_test` split. Add `--split test` or
`--split val` when the case should be restricted to one split. Case 508, for
example, is resolved as `rca_0508`; the split loader also accepts numeric case
references in the split JSON.

Each invocation creates a case-specific output root below the configured
`eval_output_dir`. The `visualization/k<N>/cases/<case>` directories retain the
initial GCP centers and volume, optimized Gaussian `.pt`/`.npz` files,
reconstructed `.npy`/NIfTI/GIF volumes, input-view and novel-view reprojection
montages, optimization timing, run metadata, and per-case metrics. The first
selected view initializes the GCP; every selected view participates in the
Gaussian primitive optimization. Add `--dry-run` to inspect the resolved plan
without starting CUDA work.

## Explicit reproduction choices

The paper does not specify the exact U-Net width, downsampling factor, offset
activation/range, depth convention, or all loss constants. This implementation
makes them checkpointed configuration values. Its defaults are a four-level
U-Net, downsampling factor 2, sigmoid ray depth, bounded tanh ZYX offsets, and
loss weights `Chamfer=1`, `soft-clDice=0.5`, and `0.01` for each of SILog,
masked depth L1, and depth-gradient L1. Every run records these choices in
`training_config.json`.
