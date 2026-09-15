"""Audit official Seeing Through Fog splits and prepare manifests, without training.

Requires the public upstream checkout; sensor downloads require DENSE registration.
Original split membership is preserved, including overlaps, and never repaired silently.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import subprocess

UPSTREAM_COMMIT = 'bb57f8c5e29a8d647d5f65b0e277401bf332cbd0'
GROUPS = {
    'train_clear': ['train_clear_day', 'train_clear_night'],
    'val_clear': ['val_clear_day', 'val_clear_night'],
    'test_clear': ['test_clear_day', 'test_clear_night'],
    'test_light_fog': ['light_fog_day', 'light_fog_night'],
    'test_dense_fog': ['dense_fog_day', 'dense_fog_night'],
    'test_snow_rain': ['snow_day', 'snow_night', 'rain'],
}


def read_split(path):
    result = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2},\d+', line):
            raise ValueError(f'{path}:{line_number}: invalid frame identifier')
        recording, frame = line.split(',')
        sample = f'{recording}_{frame}'
        if sample in seen:
            raise ValueError(f'{path}:{line_number}: duplicate frame {sample}')
        seen.add(sample)
        result.append(dict(sample_id=sample, recording=recording, frame=frame))
    if not result:
        raise ValueError(f'Empty split: {path}')
    return result


def overlap_report(groups):
    result = []
    for (left, a), (right, b) in combinations(sorted(groups.items()), 2):
        frames = sorted(set(a) & set(b))
        recordings = sorted({x.rsplit('_', 1)[0] for x in a} &
                            {x.rsplit('_', 1)[0] for x in b})
        if frames or recordings:
            result.append(dict(left=left, right=right, shared_frames=frames,
                               shared_recording_prefixes=recordings))
    return result


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, default=Path.home()/'data/seeing_through_fog_tools')
    parser.add_argument('--data', type=Path, default=Path.home()/'data/seeing_through_fog/SeeingThroughFogData')
    parser.add_argument('--archives', type=Path, default=Path.home()/'data/seeing_through_fog/downloads')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/stf_preparation')
    parser.add_argument('--verify-archives', action='store_true', help='Hash existing archive parts (can be slow)')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.tools), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f'Expected audited upstream commit {UPSTREAM_COMMIT}, got {commit}')
    if subprocess.check_output(['git', '-C', str(args.tools), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Upstream tracked files are modified; use the audited checkout')
    splits = {p.stem: read_split(p) for p in sorted((args.tools/'splits').glob('*.txt'))}
    groups = {}
    for name, components in GROUPS.items():
        groups[name] = {row['sample_id']: row for split in components for row in splits[split]}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in groups.items():
        with (args.output/f'{name}.jsonl').open('w') as stream:
            for sample in sorted(rows):
                stream.write(json.dumps(rows[sample])+'\n')
    archives = []
    for line in (args.tools/'SeeingThroughFog_sha256sum.txt').read_text().splitlines():
        expected, name = line.split()
        if Path(name).name != name or not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError('Invalid upstream checksum entry')
        path = args.archives/name
        state = 'missing'
        if path.is_file():
            state = ('verified' if sha256(path) == expected else 'checksum_mismatch') if args.verify_archives else 'present_unverified'
        archives.append(dict(name=name, sha256=expected, state=state,
                             bytes=path.stat().st_size if path.is_file() else 0))
    folders = ['cam_stereo_left_lut', 'lidar_hdl64_strongest', 'radar_targets',
               'gated0_rect', 'gated1_rect', 'gated2_rect', 'gt_labels', 'labeltool_labels']
    inventory = {name: dict(exists=(args.data/name).is_dir(),
                            entries=sum(1 for _ in (args.data/name).iterdir()) if (args.data/name).is_dir() else 0)
                 for name in folders}
    report = dict(
        state='preparation_only_not_training_ready',
        upstream='https://github.com/princeton-computational-imaging/SeeingThroughFog',
        upstream_commit=commit,
        registration='https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/dense-datasets/dense-registration-form/',
        source_split_counts={k: len(v) for k, v in splits.items()},
        source_split_sha256={p.name: sha256(p) for p in sorted((args.tools/'splits').glob('*.txt'))},
        group_counts={k: len(v) for k, v in groups.items()},
        overlaps=overlap_report(groups),
        manifest_policy='Original membership; unions remove duplicate frame IDs within a group only. Overlaps between groups are retained and reported.',
        data_root=str(args.data.resolve()), folder_inventory=inventory, archives=archives,
        free_disk_gib=shutil.disk_usage(args.data if args.data.exists() else args.tools).free/2**30,
        remaining=['Obtain registered dataset download access and check archive/extraction storage needs.',
                   'Verify archive hashes and extract required sensors, annotations and scene metadata.',
                   'Resolve frame overlaps and shared recording prefixes before defining the benchmark protocol.',
                   'Verify four-modality alignment and annotated batches on actual data.',
                   'Establish evaluator settings, then implement and test the paper-matched detector.'])
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(state=report['state'], group_counts=report['group_counts'],
                          overlaps=[dict(left=x['left'], right=x['right'], frames=len(x['shared_frames']),
                                         recordings=len(x['shared_recording_prefixes'])) for x in report['overlaps']],
                          archive_states={state:sum(x['state']==state for x in archives) for state in sorted({x['state'] for x in archives})},
                          audit=str(args.output/'audit.json')), indent=2))


if __name__ == '__main__':
    main()
