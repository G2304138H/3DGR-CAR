# Stage-2 NPZ reconstruction

`train_stage2_npz.py` reads the Stage-2 ImageCAS NPZ format directly, selects
one or more stored views, creates an ASTRA `cone_vec` geometry from their `theta_deg`
and `phi_deg`, runs FDK initialization, optimizes the 3D Gaussians for that
case, and exports the Gaussian parameters, reconstructed volume, and remaining
novel views.

The stored `images` are binary artery silhouettes, not CT line integrals.  The
script therefore compares them to `1 - exp(-gain * line_integral)` during
optimization.  The gain is estimated from the FDK initialization unless
`--silhouette-gain` is supplied.

## Example

From the `3dgs-car` directory:

```bash
python train_stage2_npz.py \
  --input /Users/renyu/Desktop/rca_0001.npz \
  --output-dir ./outputs/rca_0001_views_0_1 \
  --view-indices 0 1 \
  --iterations 8000 \
  --prediction-threshold-percentile 97
```

For a short smoke run before committing to the full optimization:

```bash
python train_stage2_npz.py \
  --input /Users/renyu/Desktop/rca_0001.npz \
  --output-dir ./outputs/rca_0001_smoke \
  --view-indices 0 1 \
  --iterations 10 \
  --num-init-gaussians 1000 \
  --no-densify \
  --record-optimization-time \
  --early-stop-checks 0
```

The source-to-isocentre distance is set to `0.75 m`, matching the hard-coded
value in the supplied Stage-2 renderer. `sid` and detector pixel spacing are
read from the NPZ. The reconstruction field of view defaults to the physical
detector width projected back to isocentre; override it with
`--volume-extent-m` when using a different crop.

## Outputs

- `gaussians.pt`: reloadable raw Gaussian state plus activated parameters and geometry.
- `gaussians.npz`: portable centres, densities, scales, and rotations.
- `initial_fbp_volume_zyx.npy`: normalized FDK initialization.
- `reconstructed_volume_zyx.npy`: final Gaussian volume in `[z,y,x]` order.
- `reconstructed_volume_xyz.nii.gz`: final volume with millimetre voxel spacing, when nibabel is installed.
- `reconstructed_volume.gif`: prediction-only density isosurface, synchronized to the training monitor's 24-frame, 5-FPS, fixed-22-degree camera and shared GT/prediction framing.
- `input_view_reprojections.npz`: final reprojections on the selected input views, their targets, raw line integrals, errors, and per-view metrics.
- `input_view_reprojections.png`: target/reprojection/error montage for the selected input views.
- `novel_views.npz`: projections, stored targets, errors, and per-view metrics at all unselected stored views by default.
- `novel_view_reprojections.png`: target/reprojection/error montage for the requested novel views.
- `optimization_timing.json`: CUDA-synchronized optimizer-loop wall time and run context, when `--record-optimization-time` is enabled.
- `run_metadata.json`: selected views and reconstruction settings.

The GIF and evaluation use the same threshold interface. The single-case
trainer defaults to an isovalue at the 97th percentile of the reconstructed
volume's strictly positive voxels. Select either
`--prediction-threshold VALUE` for a fixed raw-density threshold or
`--prediction-threshold-percentile P` for a per-case positive-voxel percentile;
the flags are mutually exclusive. The older `--volume-gif-isovalue VALUE` name
remains an alias for `--prediction-threshold VALUE`. The requested mode and the
effective numeric GIF isovalue are saved in `run_metadata.json`. Change the
monitor timing with `--monitor-gif-frames` and `--monitor-gif-fps`, or disable
this output with `--no-volume-gif`.

## Optimization timing

Pass `--record-optimization-time` to measure the complete Gaussian optimization
loop with CUDA synchronization immediately before and after it. The reported
wall time includes all iterations, ASTRA forward/backprojections, backward
passes, optimizer steps, logging checks, best-state copies, and optional
densification performed inside the loop. It excludes FDK/BP initialization,
final-volume generation, reprojections, GIF rendering, and file output.

The console reports total seconds and seconds per completed iteration, and the
same values are saved in `optimization_timing.json` and `run_metadata.json`.
Keep the iteration count, early-stopping settings, volume size, Gaussian count,
densification settings, input views, and logging interval equal when comparing
methods.

ASTRA 2.4 or newer is strongly recommended because its direct projector API
accepts PyTorch CUDA tensors. Older ASTRA versions use the implemented
CPU-transfer fallback at each optimization iteration and will be much slower.

## Split evaluation against ground-truth volumes

`evaluate_stage2_npz.py` runs the single-case reconstruction over one split and
computes voxel metrics against a second directory of case-matched ground-truth
NPZ files. The split JSON may contain top-level `train`, `validation`/`val`, and
`test` lists, those keys may be nested under `splits`, and each list item may be
a case name/number or a record such as `{"case_name": "rca_0001"}`.

Ground-truth files may use the ImageCAS layout
`<ground-truth-dir>/<vessel_type>/<case_id>.npz`, for example `lca/1.npz` and
`rca/1.npz`. The evaluator reads `vessel_type` and `case_id` from each projection
NPZ and tries this relative path first, so LCA and RCA cases with the same
numeric ID are unambiguous. A vessel-specific directory such as
`--ground-truth-dir /home/renyu/data/imagecas_voxel/lca` is also supported.

```bash
python evaluate_stage2_npz.py \
  --input-dir /path/to/projection_npzs \
  --split-json /path/to/case_split.json \
  --split test \
  --ground-truth-dir /path/to/volume_npzs \
  --output-dir ./outputs/stage2_test \
  --view-indices 0 1 \
  --iterations 8000 \
  --log-every 100 \
  --early-stop-checks 7 \
  --prediction-threshold-percentile 97
```

`--view-indices` accepts one or more indices in both the single-case trainer and
split evaluator; the evaluator applies that selected list to every case in the
split. Options not recognised by the evaluator, such as `--iterations` and
other optimization settings, are forwarded to `train_stage2_npz.py`. Use
`--reuse-existing` to keep already completed case
reconstructions, or `--skip-reconstruction` to evaluate existing
`cases/<case>/reconstructed_volume_zyx.npy` files without CUDA.

Split evaluation enforces early stopping for every optimized case.
`--early-stop-checks` must be positive and defaults to 7. With
`--log-every 100`, a case stops after seven consecutive 100-iteration logging
checks fail to improve its best checked loss. The effective setting is passed
to every `train_stage2_npz.py` subprocess and recorded in
`metrics/evaluation_config.json`.

The ground-truth volume key is auto-detected from common names including `vol`,
`volume`, `voxel`, `gt_volume`, `segmentation`, and `mask`. For the supplied
ImageCAS format, `vol` is interpreted as `[x,y,z]`, transposed internally to
`[z,y,x]`, and its `spacing` values are interpreted as XYZ millimetres. Other
volume keys default to ZYX. Override these choices with
`--ground-truth-axis-order`, `--ground-truth-spacing-key`, or
`--ground-truth-spacing-units`.

### Reversing the projection-centering offset

By default, evaluation reads `projection_center_offset` from each projection
NPZ and reverses it before computing any voxel metric. The Stage-2 renderer
centres artery coordinates as
`centered_xyz = original_xyz - projection_center_offset_xyz`. For a full-size
GT mask with physical spacing, evaluation samples the GT at
`original_xyz = prediction_local_xyz + projection_center_offset_xyz`, producing
a GT mask on the prediction's grid. This physical crop/resampling converts an
ImageCAS mask such as XYZ `(512,512,275)` into the prediction's ZYX grid, such
as `(128,128,128)`, without shape-only resizing or clipping the prediction.

The prediction grid uses `volume_extent_m / (N - 1)` because the trainer creates
it with `linspace(0, 1, N)`, including both physical endpoints. The extent is
read from each case's `run_metadata.json`. When scoring an older standalone
reconstruction without that file, pass `--evaluation-volume-extent-m VALUE`.
Binary GT is sampled with nearest-neighbour interpolation by default; change it
with `--ground-truth-interpolation linear`.

The attached `artery_mask.npz` has no NIfTI affine, origin, or direction matrix,
so evaluation assumes GT array index `[0,0,0]` is physical XYZ `(0,0,0)` metres
and all axes increase positively. If the mask converter uses a different
axis-aligned physical frame, set `--ground-truth-origin-m X Y Z` and
`--ground-truth-direction-signs SX SY SZ`. These alignment settings affect the
metrics materially and should match the source NIfTI convention.

Use `--projection-offset-mode required` to fail if a case lacks the offset, or
`--projection-offset-mode ignore` only for already aligned data. Each case JSON
records the stored XYZ offset, GT spacing/origin/direction, original and aligned
shapes, interpolation, and physical-extent source. `evaluation_arrays.npz`
contains the prediction and physically aligned GT volume and masks actually
used for Dice, MSE, and SSIM. The original reconstruction remains unchanged in
`reconstructed_volume_zyx.npy`.

### JSON-only output and timing

Split evaluation defaults to `--output-mode json-only`. During each case, the
trainer uses `--evaluation-cache-only`: it creates only the reconstructed NPY,
CUDA-synchronized optimization timing JSON, and run metadata required for
evaluation. It does not create Gaussian checkpoints, portable NPZs, NIfTI,
reprojection NPZ/PNG files, novel-view outputs, or GIFs. After metrics and
timing are captured, the evaluator removes that newly created case cache.

The only persistent output in a fresh output directory is
`evaluation_results.json`. It contains:

- the resolved configuration and effective trainer arguments;
- every case's metrics, status, alignment metadata, early-stopping result, and timing;
- the case-by-metric matrix as JSON arrays (`case_names`, `metric_names`, and `values`);
- aggregate metric mean/std/standard-error/min/max values and finite sample counts;
- optimization-only, reconstruction-wall, metrics, and total-case timing summaries;
- average seconds per completed case and estimated full/remaining split time.

The console prints the rolling mean seconds per case and estimated remaining
hours after every completed case. Pass `--keep-case-cache` to retain temporary
case files. Pass `--output-mode full` only when the legacy CSV, NPZ matrix,
per-case arrays/JSON, reconstruction artifacts, PNGs, and other diagnostic
outputs are wanted.

Reported scalars include masked 3D Dice, full-volume 3D MSE and SSIM, and
masked MSE/MAE/PSNR/SSIM. Dice compares thresholded prediction and GT masks;
by default, each prediction uses P97 of its strictly positive raw voxel values,
the same per-case isovalue rule used by `reconstructed_volume.gif`. The final
JSON records both P97 and the resulting numeric threshold for every case. The
binary GT mask defaults to `GT > 0` (`GT > 0.5` is equivalent for `{0,1}` GT).
Use `--prediction-threshold VALUE` to replace P97 with an absolute threshold on
the raw prediction, or `--prediction-threshold-percentile P` to choose a
different positive-voxel percentile. Use `--ground-truth-threshold` to change
the GT threshold. By default, other masked metrics use GT foreground voxels;
`--metric-mask union` or `--metric-mask all` changes that region. If a GT NPZ
contains a separate ROI, `--evaluation-mask-key <key>` restricts all mask
comparisons to that ROI.

FDK is mathematically designed for a circular, densely sampled cone-beam scan.
Sparse Stage-2 clinical views are neither dense nor generally co-circular, so the
FDK result is only a rough Gaussian initialization and will contain artifacts.
Use `--init-method bp` only if the installed ASTRA version rejects FDK for the
small set of cone-vector views.

## Minimal Python environment (CUDA 12.4 driver)

The Stage-2 path does not require ODL or a locally installed CUDA compiler. If
`simple-knn` is unavailable, Gaussian initialization uses a one-time chunked
PyTorch nearest-neighbour calculation instead.

```bash
python3 -m venv /home/renyu/vir_env/3dgr-car
source /home/renyu/vir_env/3dgr-car/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
cd /path/to/3DGR-CAR/3dgs-car
python -m pip install -r requirements-stage2.txt
```

Verify the GPU packages with:

```bash
python - <<'PY'
import astra
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("ASTRA:", astra.__version__)
PY
```
