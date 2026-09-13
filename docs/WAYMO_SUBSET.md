# sensor-fusion

3D object detection experiments on the Waymo Open Dataset and nuScenes.

Datasets are stored outside this repository, under `~/data/`.

## Waymo Open Dataset (Perception)

> The Perception dataset, with high resolution sensor data and labels for various tasks.

Perception v1.4.3 (with maps), in segment `.tfrecord` format. A small subset is
stored locally for pipeline testing — five training segments and two validation
segments, 6.4 GiB total. This is not a representative benchmark.

```text
~/data/waymo/
├── training/      # 5 .tfrecord segments
└── validation/    # 2 .tfrecord segments
```

## nuScenes

> nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D
> bounding boxes for 23 classes and 8 attributes, recorded with 6 cameras,
> 5 radars and 1 lidar, all with full 360 degree field of view.

The `v1.0-mini` split is stored locally — 10 scenes drawn from the full trainval
set, carrying the same sensor suite and annotation format, used as a fast
correctness check before scaling up.

```text
~/data/nuscenes/
├── maps/
├── samples/       # Sensor data for keyframes
├── sweeps/        # Sensor data for intermediate frames
└── v1.0-mini/     # JSON metadata tables
```

The full trainval blobs are gated behind a nuScenes account and are downloaded
separately; they unpack into the same directory so the folders merge.
