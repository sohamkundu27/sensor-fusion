"""Train the entropy-gated 3D baseline; supports bounded benchmarks and resume."""
import argparse
from collections import deque
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import NuScenesFusionDataset, collate_fusion_batch, CAMERAS
from models.detector import EntropyFusionDetector
from models.losses import detection_loss

INPUT_KEYS = ('camera', 'lidar', 'radar', 'camera_mask', 'lidar_mask', 'radar_mask')


def move_inputs(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in INPUT_KEYS}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=str(Path(__file__).parent/'configs/mini.json'))
    known, _ = pre.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument('--root', default=str(Path.home()/'data/nuscenes'))
    parser.add_argument('--version', choices=['v1.0-mini', 'v1.0-trainval'])
    parser.add_argument('--split', choices=['mini_train', 'train'])
    parser.add_argument('--cameras', nargs='+', choices=CAMERAS)
    parser.add_argument('--image-hw', nargs=2, type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--modality-dropout', type=float)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-steps', type=int, default=0, help='Stop at this absolute microbatch step; 0 runs requested epochs')
    parser.add_argument('--checkpoint-every', type=int, default=0, help='Save every N microbatches at optimizer boundaries; 0 saves at epoch/stop only')
    parser.add_argument('--output', type=Path, default=Path('outputs/train_mini'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-pretrained', action='store_true', help='Explicit random-camera ablation; default downloads ImageNet weights')
    parser.set_defaults(**json.loads(Path(known.config).read_text()))
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.accumulation_steps) < 1 or args.workers < 0 or args.max_steps < 0 or args.checkpoint_every < 0:
        parser.error('Positive batch/epoch/accumulation values and nonnegative workers/max steps required')
    if args.lr <= 0 or not 0 <= args.modality_dropout <= 1:
        parser.error('Positive LR and dropout in [0, 1] required')
    if args.amp and not args.device.startswith('cuda'):
        parser.error('This runner supports AMP on CUDA only')
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'last.pt').exists() and args.resume is None:
        raise FileExistsError('Output already contains a checkpoint. Use --resume or a new --output.')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint.get('format_version') != 1:
            raise ValueError('Checkpoint predates the camera-relative 3D encoding')
        for key in ('version', 'split', 'cameras', 'image_hw', 'batch_size', 'lr', 'amp', 'seed', 'accumulation_steps', 'modality_dropout'):
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume setting differs: {key}. Use the original training settings.')
    dataset = NuScenesFusionDataset(args.root, version=args.version, split=args.split,
                                   cameras=args.cameras, image_hw=args.image_hw)
    model = EntropyFusionDetector(pretrained=not args.no_pretrained and checkpoint is None,
                                  modality_dropout=args.modality_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp, init_scale=1024.)
    step, start_epoch, cursor = 0, 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step, start_epoch, cursor = checkpoint['step'], checkpoint['epoch'], checkpoint['batch_cursor']
        random.setstate(checkpoint['rng_python'])
        np.random.set_state(checkpoint['rng_numpy'])
        torch.set_rng_state(checkpoint['rng_torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    pretrained_camera = checkpoint['pretrained_camera'] if checkpoint else not args.no_pretrained
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    model.train()
    rows = deque(maxlen=1000)
    run_steps = 0
    measured_seconds = 0.
    measured_count = 0
    consecutive_overflows = 0
    parameters = sum(p.numel() for p in model.parameters())
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def save(epoch, batch_cursor):
        payload = dict(format_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                       config=config, epoch=epoch, batch_cursor=batch_cursor, step=step,
                       pretrained_camera=pretrained_camera, rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                       rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        temporary = args.output/'last.tmp'
        torch.save(payload, temporary)
        temporary.replace(args.output/'last.pt')
    for epoch in range(start_epoch, args.epochs):
        if args.max_steps and step >= args.max_steps:
            break
        # A dedicated, epoch-seeded generator makes shuffle order reproducible on resume.
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=collate_fusion_batch, pin_memory=device.type=='cuda',
                            generator=torch.Generator().manual_seed(args.seed+epoch))
        iterator = iter(loader)
        first = cursor if epoch == start_epoch else 0
        for _ in range(first):
            next(iterator)
        optimizer.zero_grad(set_to_none=True)
        window = 0
        for batch_idx in range(first, len(loader)):
            sync()
            started = time.perf_counter()
            batch = next(iterator)
            inputs = move_inputs(batch, device)
            sync()
            loaded = time.perf_counter()
            if window == 0:
                window_size = min(args.accumulation_steps, len(loader)-batch_idx,
                                  args.max_steps-step if args.max_steps else len(loader))
            with torch.autocast(device_type=device.type, enabled=args.amp, dtype=torch.float16):
                pred = model(inputs)
            losses = detection_loss(pred, batch['targets'], batch['metadata'], inputs['camera_mask'])
            if not torch.isfinite(losses['total']):
                raise FloatingPointError('Nonfinite loss; no new checkpoint written')
            scaler.scale(losses['total']/window_size).backward()
            window += 1
            gradient_norm = None
            skipped_optimizer_step = False
            if window == window_size:
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in gradients]))
                if torch.isfinite(norm):
                    gradient_norm = float(norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                    consecutive_overflows = 0
                elif args.amp:
                    # GradScaler recorded overflow during unscale: step skips the update.
                    skipped_optimizer_step = True
                    consecutive_overflows += 1
                    if consecutive_overflows > 10:
                        raise FloatingPointError('Repeated AMP overflow; use FP32 or inspect the losses')
                else:
                    raise FloatingPointError('Nonfinite gradients')
                del gradients, norm
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                window = 0
            sync()
            ended = time.perf_counter()
            step += 1
            row = dict(step=step, epoch=epoch+1, data_seconds=loaded-started, compute_seconds=ended-loaded,
                       seconds=ended-started, gradient_norm=gradient_norm, skipped_optimizer_step=skipped_optimizer_step,
                       losses={k: float(v.detach()) if torch.is_tensor(v) else v for k,v in losses.items()},
                       active_modalities=pred['available'].sum(0).tolist(),
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else 0)
            rows.append(row)
            run_steps += 1
            if run_steps > 2:
                measured_seconds += row['seconds']
                measured_count += 1
            print(json.dumps(row), flush=True)
            with (args.output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if args.checkpoint_every and step % args.checkpoint_every == 0 and window == 0:
                save(epoch+1 if batch_idx+1 == len(loader) else epoch, 0 if batch_idx+1 == len(loader) else batch_idx+1)
            if args.max_steps and step >= args.max_steps:
                save(epoch+1 if batch_idx+1 == len(loader) else epoch, 0 if batch_idx+1 == len(loader) else batch_idx+1)
                break
        else:
            save(epoch+1, 0)
    measured = list(rows)[2:] if len(rows) > 2 else list(rows)
    summary = dict(steps_completed_this_run=run_steps, total_steps=step, parameters=parameters, dataset_items=len(dataset),
                   mean_seconds=measured_seconds/measured_count if measured_count else (sum(x['seconds'] for x in measured)/len(measured) if measured else None),
                   peak_allocated_gib=max((x['peak_allocated_gib'] for x in rows), default=0),
                   peak_reserved_gib=max((x['peak_reserved_gib'] for x in rows), default=0),
                   pretrained_camera=pretrained_camera, checkpoint=str(args.output/'last.pt'), config=config)
    (args.output/'report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
