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
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Training configuration; stored in checkpoints')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-if-present', action='store_true', help='Start fresh or resume a checkpoint in this experiment directory')
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted experiment and retain validation history')
    parser.add_argument('--root', default=str(Path.home()/'data/nuscenes'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--validate-every', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1 or args.validate_every < 1:
        parser.error('Epoch and validation intervals must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.resume = args.resume or (args.resume_if_present and (output/'training/last.pt').exists())
    if any(output.iterdir()) and not args.resume:
        raise FileExistsError('Choose an empty experiment directory or use --resume')
    status = dict(state='starting', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  epochs=args.epochs, validation=[], output=str(output))
    elapsed_offset = 0.
    resume_epoch, resume_cursor = 0, 0
    if args.resume:
        old = json.loads((output/'status.json').read_text())
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Previous runner PID is still alive; stop it before resuming')
        if old['epochs'] != args.epochs:
            raise ValueError('Resume requires the original epoch target')
        checkpoint = torch.load(output/'training/last.pt', map_location='cpu', weights_only=False)
        if checkpoint['config']['version'] != 'v1.0-trainval' or Path(checkpoint['config']['root']).resolve() != Path(args.root).resolve():
            raise ValueError('Checkpoint dataset differs from the requested dataset')
        step = checkpoint['step']
        resume_epoch, resume_cursor = checkpoint['epoch'], checkpoint['batch_cursor']
        del checkpoint
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copy2(output/'status.json', output/f'status_before_resume_{stamp}.json')
        metrics = output/'training/metrics.jsonl'
        if metrics.exists():
            shutil.copy2(metrics, output/'training'/f'metrics_before_resume_{stamp}.jsonl')
            temporary = metrics.with_suffix('.tmp')
            with metrics.open() as source, temporary.open('w') as destination:
                for line in source:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # The original bytes remain in the archived log.
                    if row['step'] <= step:
                        destination.write(line)
            temporary.replace(metrics)
        status = old
        status.setdefault('resumes', []).append(dict(utc=datetime.now(timezone.utc).isoformat(), step=step,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
        status.update(pid=os.getpid(), state='resuming')
        status.pop('error', None)
        elapsed_offset = old.get('elapsed_seconds', 0.)
    start = time.monotonic()
    def update(**values):
        status.update(values, elapsed_seconds=elapsed_offset+time.monotonic()-start)
        temporary = output/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'status.json')
        print(json.dumps(status, allow_nan=False), flush=True)
    def run(command, log):
        with log.open('a') as stream:
            stream.write('\nRUN '+datetime.now(timezone.utc).isoformat()+' '+repr(command)+'\n')
            stream.flush()
            subprocess.run([sys.executable, '-u', *command], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    stages = sorted({1, args.epochs, *range(args.validate_every, args.epochs+1, args.validate_every)})
    best = max((v['mAP'] for v in status['validation']), default=-1.)
    try:
        for stage in stages:
            if any(v['epoch'] == stage for v in status['validation']):
                continue
            historical = resume_epoch > stage or (resume_epoch == stage and resume_cursor > 0)
            checkpoint = output/f'epoch_{stage:02}.pt'
            if historical and not checkpoint.exists():
                raise RuntimeError(f'Missing historical checkpoint for unevaluated epoch {stage}')
            update(state='training', target_epoch=stage)
            command = ['train.py', '--device', 'cuda', '--epochs', str(stage), '--output', str(output/'training'), '--version', 'v1.0-trainval', '--split', 'train', '--root', args.root, '--checkpoint-every', '1000']
            if args.config:
                command += ['--config', str(args.config.resolve())]
            if (output/'training/last.pt').exists():
                command += ['--resume', str(output/'training/last.pt')]
            if not historical:
                run(command, output/f'train_to_epoch_{stage:02}.log')
                shutil.copy2(output/'training/last.pt', checkpoint)
            update(state='evaluating', completed_epochs=stage)
            eval_dir = output/f'eval_epoch_{stage:02}'
            run(['eval.py', '--device', 'cuda', '--root', args.root, '--split', 'val', '--checkpoint', str(checkpoint), '--output', str(eval_dir)], output/f'eval_epoch_{stage:02}.log')
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
