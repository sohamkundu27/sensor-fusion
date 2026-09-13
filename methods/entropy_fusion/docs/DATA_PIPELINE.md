# nuScenes-mini data contract

`NuScenesFusionDataset` yields one **keyframe / camera-view** item. It defaults to
`v1.0-mini`, `mini_train`, and `CAM_FRONT`; all six cameras are supported. Official
scene lists define the split before camera expansion (8 training scenes / 2
validation scenes), preventing views of a scene from crossing splits.

## Sensor geometry

For each LiDAR or radar sample_data record, the transform is:

```text
sensor coordinates -> ego at sensor timestamp -> global
                   -> ego at camera timestamp -> camera coordinates
```

Both calibrated-sensor extrinsics and the individual sample_data ego poses are
used. The camera pinhole intrinsics project positive camera-Z points to pixels.
This follows the [official devkit projection](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py).
Ego motion is compensated; independent moving-object motion between timestamps
is not. Per-sensor time offsets and transforms are returned for inspection.

- One current LiDAR keyframe; all five current radar keyframes are combined.
  No temporal sweep accumulation is used at this milestone.
- Radar keeps the devkit's default valid/ambiguity/dynamic-state filters, passed
  explicitly without modifying global devkit settings. Radar height is sparse
  and imprecise: no artificial vertical expansion or depth completion is applied.
- Keep points with `1 < camera_Z <= 100` metres, finite coordinates, and pixels
  inside the original image. Defaults can be overridden in the dataset constructor.
- Rasterize **after** projection/resizing. Use floor pixel coordinates and retain
  the nearest positive depth when points collide. Missing pixels are zero with a
  separate boolean validity mask; no-return pixels are not measured zero-depth.
- RGB is letterboxed to 640×384 (width×height), preserving aspect ratio. The actual
  resize and padding transform is applied to projections, boxes, and intrinsics.
  Dimensions are divisible by the proposed 16×16 entropy patches.

## Item fields

| Field | Shape / meaning |
| --- | --- |
| `camera` | Float32 `[3,H,W]`, RGB in `[0,1]`; ImageNet normalization is deferred |
| `lidar`, `radar` | Float32 `[1,H,W]`, camera-Z metres divided by max_depth |
| `camera_mask` | Bool `[1,H,W]`, valid image region excluding padding |
| `lidar_mask`, `radar_mask` | Bool `[1,H,W]`, actual sensor returns |
| `lidar_depth_m`, `radar_depth_m` | Float32 `[1,H,W]`, unnormalized camera-Z metres |
| `target.boxes` | Float32 `[N,4]`, resized-image xyxy pixel boxes |
| `target.labels` | Int64 `[N]`, classes 1–10; 0 reserved for background |
| `target.annotations_3d` | Matched original global translation, w/l/h size, quaternion, attributes, visibility, point counts, velocity + validity flag |
| `metadata` | Sample/scene/camera tokens, intrinsics, transforms, original size, timestamps |
| `projections` | Variable-length visible `[N,3]` arrays of resized `(u,v,depth)` points |

The 2D targets are projections of annotated **3D boxes**, not hand-labeled 2D
occlusion boundaries. Boxes crossing the camera near plane are clipped before
projection; the convex hull is intersected with the original image. Unsupported
categories and boxes outside the view are excluded. Annotation visibility is
retained, without additional occlusion filtering. An unavailable velocity may
contain NaNs in annotation metadata and is marked `velocity_valid=False`; it is
not a ready-to-use regression target without masking.

Class order is in `data.CLASSES`: car, truck, bus, trailer, construction_vehicle,
pedestrian, motorcycle, bicycle, traffic_cone, barrier.

`collate_fusion_batch` stacks fixed-size image/map/mask tensors and returns lists
for variable-length targets, projections, and metadata. DataLoader worker processes
load CPU data; the verification script separately tests transfer to CUDA.

## Verification

```bash
source ~/venvs/entropy-fusion/bin/activate
python -m pytest -q
python -m scripts.verify_batch --workers 2
python -m scripts.verify_batch --split mini_val --batch-size 6 --workers 2 \
  --cameras CAM_BACK CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK_LEFT CAM_BACK_RIGHT \
  --output outputs/mini_val_batch
```

The verifier reads exactly one batch per invocation, checks range/shape/mask/box
invariants and split separation, compares first-item projections for all six
point sensors to the official devkit, and saves `report.json` plus
`projection_overlay.png`. Reference comparison excludes the devkit's 1px image
border and matches the loader's max-depth limit. Tolerances are 0.05 original-image
pixels and 2 mm depth: the devkit sequentially mutates float32 LiDAR points through
large global translations, while this loader composes transforms in float64.
Synthetic tests cover transform ordering and temporal poses, behind-camera points,
nearest-depth collisions, empty sensors, and near-plane box clipping.

Reports/overlays stay in ignored `outputs/`; no dataset files are copied or changed.
A passing batch is a geometry/data-loading smoke test, not detection accuracy or
a guarantee that the full dataset has finished downloading.
