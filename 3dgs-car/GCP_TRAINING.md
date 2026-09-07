# Monocular Gaussian Center Predictor

This implementation adds the learned initialization stage described by
3DGR-CAR while retaining the existing FDK/BP baselines. Each stored projection
is one monocular GCP training sample. During reconstruction, only the first
selected view initializes the Gaussians; all selected views are used by the
subsequent projection optimization.

## Paired data

The projection NPZ must follow the Stage-2 contract documented in
`STAGE2_NPZ_RECONSTRUCTION.md`. In particular, it supplies `images`,
`theta_deg`, `phi_deg`, `sid`, `imager_pixel_spacing`, and optionally
`projection_center_offset`, `vessel_type`, and `case_id`.

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
records with a case-name/id field.

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

## Explicit reproduction choices

The paper does not specify the exact U-Net width, downsampling factor, offset
activation/range, depth convention, or all loss constants. This implementation
makes them checkpointed configuration values. Its defaults are a four-level
U-Net, downsampling factor 2, sigmoid ray depth, bounded tanh ZYX offsets, and
loss weights `Chamfer=1`, `soft-clDice=0.5`, and `0.01` for each of SILog,
masked depth L1, and depth-gradient L1. Every run records these choices in
`training_config.json`.
