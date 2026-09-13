"""Run a full nuScenes experiment with scheduled official validation and best checkpoint."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', default=str(Path.home()/'data/nuscenes'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--validate-every', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1 or args.validate_every < 1:
        parser.error('Epoch and validation intervals must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Choose an empty experiment directory')
    status = dict(state='starting', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  epochs=args.epochs, validation=[], output=str(output))
    start = time.monotonic()
    def update(**values):
        status.update(values, elapsed_seconds=time.monotonic()-start)
        temporary = output/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'status.json')
        print(json.dumps(status, allow_nan=False), flush=True)
    def run(command, log):
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', *command], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    stages = sorted({1, args.epochs, *range(args.validate_every, args.epochs+1, args.validate_every)})
    best = -1.
    try:
        for stage in stages:
            update(state='training', target_epoch=stage)
            command = ['train.py', '--epochs', str(stage), '--output', str(output/'training'), '--version', 'v1.0-trainval', '--split', 'train', '--root', args.root, '--checkpoint-every', '1000']
            if (output/'training/last.pt').exists():
                command += ['--resume', str(output/'training/last.pt')]
            run(command, output/f'train_to_epoch_{stage:02}.log')
            checkpoint = output/f'epoch_{stage:02}.pt'
            shutil.copy2(output/'training/last.pt', checkpoint)
            update(state='evaluating', completed_epochs=stage)
            eval_dir = output/f'eval_epoch_{stage:02}'
            run(['eval.py', '--root', args.root, '--split', 'val', '--checkpoint', str(checkpoint), '--output', str(eval_dir)], output/f'eval_epoch_{stage:02}.log')
            metrics = json.loads((eval_dir/'metrics_summary.json').read_text())
            record = dict(epoch=stage, mAP=metrics['mean_ap'], NDS=metrics['nd_score'], checkpoint=str(checkpoint))
            status['validation'].append(record)
            if record['mAP'] > best:
                best = record['mAP']
                shutil.copy2(checkpoint, output/'best.pt')
                status['best_epoch'], status['best_mAP'] = stage, best
            update(state='stage_complete')
        update(state='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        update(state='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
