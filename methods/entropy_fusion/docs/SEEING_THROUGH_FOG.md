# Seeing Through Fog experiment preparation

Status on September 15, 2026: public tools and split manifests prepared; the full
dataset has not been downloaded and no STF training has started. Dataset access
requires the user's [DENSE registration](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/dense-datasets/dense-registration-form/).
The form requests affiliation/contact information and acceptance of the dataset
terms. Provide the resulting download link locally; do not commit access links.

## Why the nuScenes result is lower

The completed revised nuScenes experiment selected epoch 15 with **21.96% mAP /
28.60% NDS**, compared with 15.91% / 24.17% for the previous implementation.
This is a ten-class, camera-ray 3D adaptation with different sensors, backbones,
anchors and training settings. It does not reproduce the original experiment.

The original paper's Table 2 reports car AP under KITTI-style difficulty levels
on its own weather dataset, not nuScenes NDS or ten-class nuScenes mAP. Its
moderate car AP is 85.57% in clear weather, 87.99% in light fog, 81.49% in dense
fog and 83.71% in snow/rain. Merely using its dataset will not make those numbers
directly comparable; we must also pin the detection/evaluation protocol.

There are real weaknesses in our model, beyond that comparison mismatch. In the
best nuScenes checkpoint, car AP is 6.76%, 28.56%, 49.28% and 64.74% at 0.5, 1,
2 and 4 metre matching thresholds. This suggests precise localization remains a
major limitation. Mean translation error is 0.842m and velocity error 1.394m/s
under the evaluator's matched-detection rules. Trailer AP is only 2.56% and
construction-vehicle AP 3.12%; these class failures also reduce the ten-class
average. These are observed diagnostics, not proof of a single architectural cause.
Source: `outputs/paper_v2_full_20260914/eval_epoch_15/metrics_summary.json`.

## Public resources prepared

The [official repository](https://github.com/princeton-computational-imaging/SeeingThroughFog)
is checked out outside this project at `~/data/seeing_through_fog_tools`, pinned
to `bb57f8c5e29a8d647d5f65b0e277401bf332cbd0`, with its license intact.
It contains calibration, projection, conversion and visualization utilities and
example images. It does not contain a complete released detector trainer or a
Table-2 evaluation entry point. We have not installed its old TensorFlow/Qt
environment into the working nuScenes environment.

Dataset directories are prepared at:

- `~/data/seeing_through_fog/downloads` for archives.
- `~/data/seeing_through_fog/SeeingThroughFogData` for extracted measurements.

The upstream checksum file lists **19 archive components**, `.z01` through `.z18`
plus `.zip`; all are currently missing. About 269 GiB was free before preparation.
Compressed and extracted sizes must be checked against the actual download listing
before transfer. The upstream format involves an outer multipart archive followed
by per-sensor archives. Keep required RGB, gated NIR, LiDAR, radar, annotations and
scene metadata; do not blindly extract all optional sensor/history products.

Run this preparation audit from the method directory:

```bash
python -m scripts.prepare_stf
# After obtaining the archives, verify against the upstream hashes:
python -m scripts.prepare_stf --verify-archives
```

The script checks the upstream revision, preserves source split identifiers and
hashes, produces manifests under `outputs/stf_preparation`, inventories folders,
and reports overlaps. It never launches training or silently changes membership.
A successful audit execution means the report was written, not that data or the
training pipeline are ready. `audit.json` explicitly records that distinction.

## Split audit and unresolved protocol details

Day/night unions of the currently published split files give:

| Group | Unique frames |
|---|---:|
| Clear training | 3,526 |
| Clear validation | 808 |
| Clear test | 1,882 |
| Light fog | 1,052 |
| Dense fog | 887 |
| Snow/rain | 4,905 |

These are counts of the public files at the pinned revision, not a claim that
they exactly recreate the published paper's final evaluation set. In particular,
`rain.txt` shares **23 frames with clear training**, 96 with clear validation,
22 with clear test and 20 with light fog. The snow/rain union deduplicates frames
within that union only. The audit also reports shared recording-name prefixes
between some training and held-out groups. Shared prefixes need investigation;
they alone do not establish geographic overlap.

Before training, resolve these discrepancies using the downloaded scene metadata
and any released protocol documentation. Preserve the original lists. If we must
derive disjoint splits, publish the exact exclusions and call that a modified
benchmark, rather than silently presenting it as the paper's protocol.

The paper specifies four streams (RGB, gated NIR, LiDAR, radar), image-plane
fusion, half-width VGG features, six detection scales, 21 training-box-clustered
anchor templates, 16x16 entropy patches, CE plus Huber loss, 5:1 hard-negative
mining, clear-weather-only training, constant learning rate, training from
scratch and weight decay 0.0005. The existing nuScenes model is not a drop-in
replacement. Gated image composition, normalization, mapping/cropping and the
released homography's coordinate conventions require verification on real data.
The public depth-warping example is not automatically the planar homography
recipe described in the paper.

The exact AP interpolation, IoU thresholds, handling of ignored/ambiguous labels,
2D versus 3D output used for Table 2, input resize and all numerical training
hyperparameters must be established before claiming a matched reproduction.
The 0.5 threshold in the paper's loss section is anchor matching, and must not
be mistaken for a documented evaluation threshold. Unspecified choices will be
recorded explicitly, rather than invented as paper settings.

## Next steps when access arrives

1. Inspect the registered download listing and storage requirements; download,
   verify hashes and extract the required data products.
2. Resolve the split discrepancies and freeze the evaluation protocol.
3. Implement the four-stream dataset adapter; inspect aligned RGB/gated/LiDAR/radar
   overlays and labels, then verify a real annotated batch end to end.
4. Implement the paper-matched detector and test a small training-set overfit
   before launching a full run. Start from scratch for the reproduction.
5. Compare against an otherwise matched fusion model without entropy gates;
   select settings on clear validation and report each held-out weather group.

References: [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bijelic_Seeing_Through_Fog_Without_Seeing_Fog_Deep_Multimodal_Sensor_Fusion_CVPR_2020_paper.html),
[supplement](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/figures/AdverseWeatherFusion_Supplement.pdf),
[dataset access](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/dense-datasets/).
