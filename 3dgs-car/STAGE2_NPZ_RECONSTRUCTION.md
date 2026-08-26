# Stage-2 NPZ reconstruction

`train_stage2_npz.py` reads the Stage-2 ImageCAS NPZ format directly, selects
two stored views, creates an ASTRA `cone_vec` geometry from their `theta_deg`
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
  --iterations 8000
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
- `input_view_reprojections.npz`: final reprojections on the two selected input views, their targets, raw line integrals, errors, and per-view metrics.
- `input_view_reprojections.png`: target/reprojection/error montage for the two selected input views.
- `novel_views.npz`: projections at all unselected stored views by default.
- `optimization_timing.json`: CUDA-synchronized optimizer-loop wall time and run context, when `--record-optimization-time` is enabled.
- `run_metadata.json`: selected views and reconstruction settings.

The GIF defaults to an isovalue at 25% of the reconstructed volume's value
range. Override it with `--volume-gif-isovalue VALUE`. Change the monitor timing
with `--monitor-gif-frames` and `--monitor-gif-fps`, or disable this output with
`--no-volume-gif`.

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

FDK is mathematically designed for a circular, densely sampled cone-beam scan.
Two Stage-2 clinical views are neither dense nor generally co-circular, so the
FDK result is only a rough Gaussian initialization and will contain artifacts.
Use `--init-method bp` only if the installed ASTRA version rejects FDK for the
two cone-vector views.

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
