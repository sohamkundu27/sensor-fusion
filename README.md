# sensor-fusion

3D object detection experiments using the Waymo Open Dataset Perception dataset.

## Dataset

This setup uses **Perception v1.4.3 (with maps)** in segment `.tfrecord` format:

```text
gs://waymo_open_dataset_v_1_4_3/individual_files/
```

The bucket and access were verified on September 12, 2026. Its folders include
`training`, `validation`, `testing`, `testing_3d_camera_only_detection`, and
`domain_adaptation`. Check the [official download page](https://waymo.com/open/download/)
for version changes before downloading. The separately listed modular v2.0.1 release
uses a different format and is not the source for this subset.

Access requires a Google account that has accepted the
[Waymo Open Dataset license](https://waymo.com/open/).

## Google Cloud CLI setup

The development machine has Google Cloud SDK **584.0.0** and gsutil **5.37**,
installed using Google's [official Linux curl/archive method](https://cloud.google.com/sdk/docs/install),
not pip. The installation is at `~/.local/lib/google-cloud-sdk`; `~/.bashrc`
contains the PATH setup.

For the existing installation:

```bash
source "$HOME/.local/lib/google-cloud-sdk/path.bash.inc"
gcloud version
gsutil version

# Use the account that accepted the Waymo license.
gcloud auth login
gsutil ls gs://waymo_open_dataset_v_1_4_3/individual_files/
```

On another machine, install the CLI using the official instructions linked above.

## Local subset

Only **five training segments and two validation segments** were downloaded,
selected by sorting filenames alphabetically and taking the first five/two.
This is a small subset for testing the pipeline, not a representative benchmark.
Dataset files live outside this repository:

```text
~/data/waymo/
├── training/                # 5 .tfrecord files
├── validation/              # 2 .tfrecord files
├── subset-manifest.json     # Source URLs and expected sizes
├── remote-metadata.txt      # Google Cloud Storage object metadata
├── verification-report.json
├── verify_subset.py
└── DOWNLOAD.md
```

These supporting files were generated locally and are not included in this repository.

| Split | Files | Bytes | Size |
| --- | ---: | ---: | ---: |
| Training | 5 | 4,971,269,561 | 4.630 GiB |
| Validation | 2 | 1,850,834,701 | 1.724 GiB |
| **Total** | **7** | **6,822,104,262** | **6.354 GiB** |

Disk usage rounds to `6.4G` with `du -sh ~/data/waymo`.

### Downloaded filenames

Training:

```text
segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord
segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord
```

Validation:

```text
segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord
```

### Integrity checks

All seven files passed exact byte-size and MD5 comparisons against Google Cloud
Storage metadata after downloading. No temporary `.gstmp` files remained.
TensorFlow and `waymo-open-dataset` were not installed in the checked Python
environment, so verification did not include decoding frames.

To repeat the checksum checks on the configured development machine:

```bash
python3 "$HOME/data/waymo/verify_subset.py"
```

## Download the remaining training and validation files

The verified bucket listing contains **798 training files** (816,376,301,297 bytes)
and **202 validation files** (205,619,283,313 bytes): approximately **952 GiB**
combined. Allow additional space for preprocessing and experiment outputs.
The development drive had only about 583 GiB free before downloading the subset,
so choose a larger destination before fetching both complete splits.

The following commands download the rest and skip files whose checksums already
match. They do not delete extra local files. These full-split downloads have
**not** been run.

```bash
source "$HOME/.local/lib/google-cloud-sdk/path.bash.inc"

# Change this to a larger disk's path when needed.
WAYMO_DATA_DIR="$HOME/data/waymo"
mkdir -p "$WAYMO_DATA_DIR/training" "$WAYMO_DATA_DIR/validation"

gsutil -m rsync -c \
  gs://waymo_open_dataset_v_1_4_3/individual_files/training/ \
  "$WAYMO_DATA_DIR/training/"

gsutil -m rsync -c \
  gs://waymo_open_dataset_v_1_4_3/individual_files/validation/ \
  "$WAYMO_DATA_DIR/validation/"
```
