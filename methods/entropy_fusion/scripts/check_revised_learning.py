"""Bounded real-data mechanics check; no validation data or retained training weights."""
import json
from pathlib import Path
import torch
from data import NuScenesFusionDataset, collate_fusion_batch
from models.factory import build_model
from models.losses import detection_loss
from models.predictions import decode_predictions
from train import move_inputs, seed_everything


def main():
    torch.set_num_threads(8)
    seed_everything(42)
    assert torch.cuda.is_available()
    config = json.loads(Path('configs/paper_v2.json').read_text())
    config['modality_dropout'] = 0
    dataset = NuScenesFusionDataset(Path.home()/'data/nuscenes',sensor_encoding='dhi')
    batch = collate_fusion_batch([dataset[0],dataset[1]])
    inputs = move_inputs(batch,'cuda')
    model = build_model(config,pretrained=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.0005)
    rows, gradients = [], {}
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        losses = detection_loss(prediction,batch['targets'],batch['metadata'],inputs['camera_mask'])
        assert losses['positive_anchors'] > 0 and torch.isfinite(losses['total'])
        losses['total'].backward()
        if step == 0:
            for key in ('camera','lidar','radar','exchange','head'):
                norm = sum(float(p.grad.abs().sum()) for p in getattr(model,key).parameters() if p.grad is not None)
                assert norm > 0
                gradients[key] = norm
        torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        optimizer.step()
        rows.append({k:float(losses[k].detach()) for k in ('total','center_error_meters','velocity_mps')})
    mean = lambda key,values:sum(r[key] for r in values)/len(values)
    before, after = mean('total',rows[:5]),mean('total',rows[-5:])
    assert after < .7*before, (before,after)
    assert mean('center_error_meters',rows[-5:]) < mean('center_error_meters',rows[:5])
    model.eval()
    with torch.inference_mode():
        records = decode_predictions(model(inputs),batch['metadata'],score_threshold=.01)
    assert all(len(r)>0 for r in records)
    report = dict(status='PASS',steps=80,first_loss=before,last_loss=after,
        first_center_error_m=mean('center_error_meters',rows[:5]),last_center_error_m=mean('center_error_meters',rows[-5:]),
        gradient_l1=gradients,parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        decoded_per_view=[len(r) for r in records],
        note='Repeated two training views with dropout off; tests mechanics, not held-out accuracy.')
    output=Path('outputs/paper_v2_learning_check.json')
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
