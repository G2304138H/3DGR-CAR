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
- `input_view_reprojections.npz`: final reprojections on the two selected input views, their targets, raw line integrals, errors, and per-view metrics.
- `input_view_reprojections.png`: target/reprojection/error montage for the two selected input views.
- `novel_views.npz`: projections at all unselected stored views by default.
- `run_metadata.json`: selected views and reconstruction settings.

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
