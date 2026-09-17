# Feasibility: Deep Entropy Fusion / "Seeing Through Fog"

Repo: https://github.com/princeton-computational-imaging/SeeingThroughFog
Cloned commit: `bb57f8c5e29a8d647d5f65b0e277401bf332cbd0` (shallow, 2026-09-17)

## Result

| Field | Value |
|---|---|
| Ran | **N** |
| VRAM @ batch 1 | not measured |
| sec/iteration | not measured |
| Est. hrs/epoch (mini) | not measured |
| Est. hrs/epoch (full nuScenes) | not applicable (method does not use nuScenes) |
| Biggest blocker | **The official repo contains no model and no training code.** It is a dataset toolkit only. |

## Biggest blocker, in detail

This was picked as the lightest method to validate the workflow, but it cannot be
timed at all, because there is nothing to train. The published repository is the
*dataset benchmark and label tooling*, not the entropy-fusion detector from the
paper. The SSD-based fusion network was never released here.

Evidence, all from the cloned tree:

* 46 Python files total. None define a network, a loss, an optimizer, or a
  training loop. `grep` for `optimizer|\.backward\(|def train|model\.fit|SSD`
  matches exactly one file, `tools/CreateTFRecords/generic_tf_tools/resize.py`,
  and that match is incidental (an image-resize mode name), not training code.
* Only two files import a deep-learning framework at all
  (`tools/CreateTFRecords/generic_tf_tools/tf_records.py` and `data2example.py`),
  and both use TensorFlow purely to serialize TFRecords for dataset packaging.
* `environment.yml` is named `LabelTool` and pins python 3.7.1, matplotlib,
  opencv, pyquaternion, pyserial. **It does not install tensorflow or torch.**
  An environment intended for training would install one of them.
* The actual contents are: `DatasetViewer` (Qt/pyqtgraph GUI), `CreateTFRecords`,
  `ProjectionTools` (Lidar2RGB, Radar2RGB, Gated2RGB), `DatasetFoggification`,
  `Raw2LUTImages`, `DatasetStatisticsTools`, and `splits/`.

## Second, independent blocker: data

Even for the tooling that does exist, the local dataset directories are empty:

```
/home/soham/data/seeing_through_fog/downloads            (empty)
/home/soham/data/seeing_through_fog/SeeingThroughFogData (empty)
```

Total 12K — directory stubs only. The STF dataset is gated: it requires
registering on the Princeton dataset page to receive a time-limited download
link, then a two-stage 7z extraction. The README also notes the download page
has recurring downtime and that expired links require re-registration. This is a
manual, human-gated step; it is not scriptable from here.

Note that `/home/soham/data/seeing_through_fog_tools/` already holds an earlier
copy of this same toolkit (same `extract.sh`, `splits/`, `terms_of_use.txt`), so
the repo side of this method was already on the machine before this pass.

## What I did not do

I did not substitute nuScenes-mini for this method. The architecture consumes a
gated NIR camera, which exists in STF and has no counterpart in nuScenes, so a
nuScenes-mini timing number would not be a measurement of this method.

I did not attempt a reimplementation — out of scope for a feasibility pass, and
this project already has its own reimplementation under
`methods/entropy_fusion/` (see `docs/SEEING_THROUGH_FOG.md`), which is a
different artifact from the official repo that was asked about here.

## To unblock

1. Register at https://light.princeton.edu/datasets/automated_driving_dataset/
   and download `SeeingThroughFogCompressed.z*` (manual, human-gated).
2. Accept that a timing number still requires a detector implementation from
   somewhere other than this repo — either this project's own reimplementation,
   or a third-party port.
