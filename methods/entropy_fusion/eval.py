"""Evaluate all six camera views using the official nuScenes 3D detection devkit."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from data import NuScenesFusionDataset, collate_fusion_batch, CAMERAS
from models.detector import EntropyFusionDetector
from models.predictions import decode_predictions, merge_views
from train import move_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', default=str(Path.home()/'data/nuscenes'))
    parser.add_argument('--split', choices=['mini_val', 'val'], default='mini_val')
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--score-threshold', type=float, default=.05)
    parser.add_argument('--topk', type=int, default=100, help='Max predictions per view before global merging')
    parser.add_argument('--max-batches', type=int, default=0, help='Partial export smoke test only; disables official metrics')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.topk < 1 or not 0 <= args.score_threshold <= 1 or args.max_batches < 0:
        parser.error('Invalid batch, worker, topk, threshold or batch-limit setting')
    torch.set_num_threads(8)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = EntropyFusionDetector(pretrained=False, modality_dropout=config['modality_dropout']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = NuScenesFusionDataset(args.root, version='v1.0-mini' if args.split=='mini_val' else 'v1.0-trainval',
                                   split=args.split, cameras=CAMERAS, image_hw=config['image_hw'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_fusion_batch, pin_memory=device.type=='cuda')
    # Every split token must be present, including samples with no detections.
    results = {token: [] for token, _ in dataset.items}
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            prediction = model(move_inputs(batch, device))
            for meta, records in zip(batch['metadata'], decode_predictions(prediction, batch['metadata'], args.score_threshold, args.topk)):
                results[meta['sample_token']].extend(records)
                processed += 1
            if (batch_index+1) % 30 == 0:
                print(f'Processed {processed}/{len(dataset)} camera views', flush=True)
            if args.max_batches and batch_index+1 >= args.max_batches:
                break
    complete = processed == len(dataset)
    results = {token: merge_views(records) for token, records in results.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/('predictions.json' if complete else 'partial_predictions.json')
    payload = dict(meta=dict(use_camera=True, use_lidar=True, use_radar=True, use_map=False,
                            use_external=checkpoint['pretrained_camera']), results=results)
    path.write_text(json.dumps(payload, allow_nan=False))
    status = dict(complete=complete, processed_views=processed, total_views=len(dataset),
                  sample_tokens=len(results), prediction_count=sum(map(len, results.values())),
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_step=checkpoint['step'],
                  score_threshold=args.score_threshold, topk_per_view=args.topk, split=args.split)
    (args.output/'run.json').write_text(json.dumps(status, indent=2))
    if not complete:
        print('Partial export only; official metrics were NOT run.', flush=True)
        return
    evaluator = NuScenesEval(dataset.nusc, config=config_factory('detection_cvpr_2019'), result_path=str(path),
                             eval_set=args.split, output_dir=str(args.output), verbose=False)
    metrics = evaluator.main(plot_examples=0, render_curves=False)
    print(json.dumps(dict(mAP=metrics['mean_ap'], NDS=metrics['nd_score'], **status), indent=2), flush=True)


if __name__ == '__main__':
    main()
