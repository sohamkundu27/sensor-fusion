"""Wait for verified full KITTI downloads, then run the resumable car experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def validate_download(root, manifest):
    if manifest['state']!='verified' or manifest['train_count']!=3712 or manifest['val_count']!=3769:
        raise ValueError('Full verified 3712/3769 KITTI split required')
    train,val = manifest['subsets']['train'],manifest['subsets']['val']
    if len(train)!=3712 or len(val)!=3769 or len(set(train+val))!=7481:
        raise ValueError('Invalid or overlapping KITTI splits')
    for split,ids in [('train',train),('val',val)]:
        if (root/'ImageSets'/f'{split}.txt').read_text().splitlines()!=ids:
            raise ValueError('Split file differs from verified manifest')
    if len(manifest['archives'])!=4: raise ValueError('Expected four verified data products')
    expected = {f'training/{folder}/{sample}.{ext}' for folder,ext in
                [('image_2','png'),('velodyne','bin'),('calib','txt'),('label_2','txt')] for sample in train+val}
    found = set()
    for archive in manifest['archives']:
        for record in archive['files']:
            name = record['path']
            if name not in expected or name in found: raise ValueError('Unexpected or duplicate downloaded file')
            found.add(name)
            path = root/name
            if not path.is_file() or path.stat().st_size!=record['bytes']:
                raise ValueError(f'Verified download missing or changed size: {path}')
    if found!=expected: raise ValueError('Incomplete verified data products')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/'data/kitti_full')
    parser.add_argument('--output',type=Path,default=Path('outputs/kitti_full_20260915'))
    args = parser.parse_args()
    state_path = args.output.with_name(args.output.name+'.wait.json')
    state_path.parent.mkdir(parents=True,exist_ok=True)
    state = dict(pid=os.getpid(),state='waiting_for_verified_download',root=str(args.root),output=str(args.output),epochs=20)
    def update(**values):
        state.update(values,checked_utc=datetime.now(timezone.utc).isoformat())
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(state_path)
        print(json.dumps(state),flush=True)
    try:
        while True:
            try:
                manifest = json.loads((args.root/'subset_manifest.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                manifest = None
            progress = {}
            for path in args.root.glob('*_progress.json'):
                try: progress[path.stem] = json.loads(path.read_text())
                except json.JSONDecodeError: pass
            update(download_progress=progress)
            if manifest and manifest['state']=='failed': raise RuntimeError('KITTI downloader failed; inspect subset_manifest.json and resume the download service')
            if manifest and manifest['state']=='verified':
                validate_download(args.root,manifest)
                break
            time.sleep(30)
        update(state='running_experiment')
        subprocess.run([sys.executable,'-u','-m','scripts.run_full_experiment',
                        '--dataset','kitti','--config','configs/kitti.json','--root',str(args.root),
                        '--output',str(args.output),'--epochs','20','--validate-every','5','--resume-if-present'],check=True)
        update(state='completed')
    except BaseException as error:
        update(state='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__': main()
