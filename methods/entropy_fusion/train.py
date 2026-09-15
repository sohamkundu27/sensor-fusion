"""Train the entropy-gated 3D baseline; supports bounded benchmarks and resume."""
import argparse
import faulthandler
import signal
from collections import deque
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import NuScenesFusionDataset, collate_fusion_batch, CAMERAS
from data.kitti_dataset import KittiFusionDataset
from models.factory import build_model
from models.losses import detection_loss

INPUT_KEYS = ('camera', 'lidar', 'radar', 'camera_mask', 'lidar_mask', 'radar_mask', 'lidar_depth_m', 'radar_depth_m')


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
    parser.add_argument('--model-variant', choices=['baseline', 'paper_v2'], default='baseline')
    parser.add_argument('--dataset', choices=['nuscenes','kitti'], default='nuscenes')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--dropout-mode', choices=['single','single_available','independent'], default='independent')
    parser.add_argument('--center-warmup-steps', type=int, default=0)
    parser.add_argument('--fusion-mode', choices=['entropy', 'concat'], default='entropy')
    parser.add_argument('--sensor-encoding', choices=['depth', 'dhi'], default='depth')
    parser.add_argument('--schedule-epochs', type=int, default=20)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--photometric-augmentation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--worker-start-method', choices=['fork','spawn'], default='fork')
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
    if args.model_variant == 'baseline' and args.sensor_encoding != 'depth':
        parser.error('baseline requires --sensor-encoding depth')
    if args.model_variant == 'paper_v2' and args.sensor_encoding != 'dhi':
        parser.error('paper_v2 requires --sensor-encoding dhi')
    if args.dataset == 'kitti' and (args.model_variant != 'paper_v2' or args.num_classes != 1 or args.version is not None or args.split != 'train'):
        parser.error('KITTI requires paper_v2, num_classes=1, version=null and split=train; use configs/kitti.json')
    if args.dataset == 'nuscenes' and args.num_classes != 10:
        parser.error('nuScenes requires 10 foreground classes')
    if (args.model_variant == 'paper_v2' and args.schedule_epochs < args.epochs) or args.warmup_steps < 0 or args.weight_decay < 0:
        parser.error('Schedule must cover training, with nonnegative warmup/weight decay')
    return args


def main():
    # Allow thread diagnostics on a live Linux job without debugger attachment.
    faulthandler.enable()
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
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
        for key, default in [('dataset','nuscenes'), ('num_classes',10)]:
            if checkpoint['config'].get(key, default) != config[key]:
                raise ValueError(f'Resume setting differs: {key}')
        for key, default in [('model_variant','baseline'), ('fusion_mode','entropy'), ('sensor_encoding','depth'), ('weight_decay',.01), ('schedule_epochs',20), ('warmup_steps',1000), ('dropout_mode','independent'), ('center_warmup_steps',0), ('worker_start_method','fork'), ('photometric_augmentation',False)]:
            if checkpoint['config'].get(key, default) != config[key]:
                raise ValueError(f'Resume setting differs: {key}')
        for key in ('version', 'split', 'cameras', 'image_hw', 'batch_size', 'lr', 'amp', 'seed', 'accumulation_steps', 'modality_dropout'):
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume setting differs: {key}. Use the original training settings.')
    dataset = (KittiFusionDataset(args.root, split=args.split, image_hw=args.image_hw) if args.dataset == 'kitti' else
               NuScenesFusionDataset(args.root, version=args.version, split=args.split,
                                     cameras=args.cameras, image_hw=args.image_hw, sensor_encoding=args.sensor_encoding))
    model = build_model(config, pretrained=not args.no_pretrained and checkpoint is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                            generator=torch.Generator().manual_seed(args.seed+epoch),
                            timeout=120 if args.workers else 0,
                            **({'multiprocessing_context': args.worker_start_method} if args.workers else {}))
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
            if args.photometric_augmentation:
                rgb = inputs['camera']
                brightness = torch.empty(len(rgb),1,1,1,device=device).uniform_(.8,1.2)
                contrast = torch.empty_like(brightness).uniform_(.8,1.2)
                inputs['camera'] = ((rgb-.5)*contrast+.5).mul(brightness).clamp(0,1)*inputs['camera_mask']
            if args.model_variant == 'paper_v2':
                total = len(loader)*args.schedule_epochs
                warmup = max(args.warmup_steps,1)
                factor = min(1.,(step+1)/warmup)
                progress = min(1.,max(0.,(step-args.warmup_steps)/max(1,total-args.warmup_steps)))
                factor *= .05+.95*.5*(1+math.cos(math.pi*progress))
                for group in optimizer.param_groups:
                    group['lr'] = args.lr*factor
            sync()
            loaded = time.perf_counter()
            if window == 0:
                window_size = min(args.accumulation_steps, len(loader)-batch_idx,
                                  args.max_steps-step if args.max_steps else len(loader))
            with torch.autocast(device_type=device.type, enabled=args.amp, dtype=torch.float16):
                pred = model(inputs)
            if args.model_variant == 'paper_v2' and args.center_warmup_steps:
                pred['metric_center_weight'] = .05+.2*min(1.,step/args.center_warmup_steps)
            losses = detection_loss(pred, batch['targets'], batch['metadata'], inputs['camera_mask'])
            if not torch.isfinite(losses['total']):
                def stats(tensor):
                    value = tensor.detach().float()
                    finite = torch.isfinite(value)
                    return dict(nonfinite=int((~finite).sum()),
                                max_abs=float(value[finite].abs().max()) if finite.any() else None)
                failure = dict(step=step+1, samples=[dict(token=m['sample_token'], camera=m['camera_channel']) for m in batch['metadata']],
                    losses={k: str(float(v.detach())) if torch.is_tensor(v) else v for k,v in losses.items()},
                    predictions={k:stats(v) for k,v in pred.items() if torch.is_tensor(v)},
                    inputs={k:stats(v) for k,v in inputs.items()})
                (args.output/f'failure_step_{step+1:08}.json').write_text(json.dumps(failure,indent=2)+'\n')
                torch.save(dict(model=model.state_dict(), config=config, batch=batch,
                    inputs={k:v.detach().cpu() for k,v in inputs.items()},
                    available=pred['available'].detach().cpu()),args.output/f'failure_step_{step+1:08}.pt')
                print(json.dumps(failure),flush=True)
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
            row = dict(step=step, epoch=epoch+1, lr=optimizer.param_groups[0]['lr'], data_seconds=loaded-started, compute_seconds=ended-loaded,
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
