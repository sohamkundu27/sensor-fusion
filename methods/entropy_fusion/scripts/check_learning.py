"""Small real-data overfit check: verify gradients reach every stream and gate."""
import json
from pathlib import Path
import numpy as np
import torch
from data import NuScenesFusionDataset, collate_fusion_batch
from models.detector import EntropyFusionDetector
from models.losses import detection_loss
from train import move_inputs, seed_everything


def main():
    torch.set_num_threads(8)
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NuScenesFusionDataset(Path.home()/'data/nuscenes')
    batch = collate_fusion_batch([dataset[0]])
    model = EntropyFusionDetector(pretrained=True, modality_dropout=0).to(device).train()
    inputs = move_inputs(batch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses, gradients = [], {}
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        result = detection_loss(prediction, batch['targets'], batch['metadata'], inputs['camera_mask'])
        assert result['positive_anchors'] > 0
        result['total'].backward()
        if step == 0:
            for name in ('camera', 'lidar', 'radar', 'exchange', 'head'):
                module = getattr(model, name)
                norm = sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)
                assert np.isfinite(norm) and norm > 0, (name, norm)
                gradients[name] = norm
            for level, exchange in enumerate(model.exchange):
                for stream, gate in enumerate(exchange.gates):
                    norm = float(gate.weight.grad.abs().sum())
                    assert np.isfinite(norm) and norm > 0
                    gradients[f'gate_{level}_{stream}'] = norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(result['total'].detach()))
    before, after = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    assert after < .85*before, (before, after)
    report = dict(status='PASS', steps=40, first_five_mean_loss=before, last_five_mean_loss=after,
                  initial_gradient_l1=gradients, losses=losses,
                  note='One repeated training view, dropout disabled; checks learning mechanics only. Weights discarded.')
    path = Path('outputs/learning_check/report.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
