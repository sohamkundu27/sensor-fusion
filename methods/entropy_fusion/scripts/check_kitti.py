"""Verify KITTI inputs first, then run a bounded two-training-image learning check."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from data.kitti_dataset import KittiFusionDataset
from data import collate_fusion_batch
from models.factory import build_model
from models.losses import detection_loss
from train import move_inputs, seed_everything


def main():
    root = Path.home()/'data/kitti'
    output = Path('outputs/kitti_precheck')
    output.mkdir(parents=True,exist_ok=True)
    datasets = {split:KittiFusionDataset(root,split) for split in ('train','val')}
    assert not set(datasets['train'].ids)&set(datasets['val'].ids)
    selected, counts = [], {}
    for split,dataset in datasets.items():
        points, boxes = 0,0
        for index in range(len(dataset)):
            item = dataset[index]
            for key in ('camera','lidar','radar','lidar_depth_m','radar_depth_m'):
                assert torch.isfinite(item[key]).all(),(split,index,key)
            assert item['lidar_mask'].any() and not item['radar_mask'].any()
            assert not item['lidar'][:,~item['camera_mask'][0]].any()
            points += item['metadata']['projected_lidar_points']
            boxes += len(item['target']['boxes'])
            if split=='train' and len(selected)<2 and len(item['target']['boxes']): selected.append(item)
            if index==0:
                image = Image.fromarray((item['camera'].numpy().transpose(1,2,0)*255).astype(np.uint8))
                draw = ImageDraw.Draw(image)
                for u,v,depth in item['projections']['lidar'].numpy()[::5]:
                    draw.point((int(u),int(v)),fill=(int(255*(1-min(depth/80,1))),64,int(255*min(depth/80,1))))
                for box in item['target']['boxes'].tolist(): draw.rectangle(box,outline='lime',width=2)
                image.save(output/f'{split}_alignment.png')
        counts[split] = dict(frames=len(dataset),car_boxes=boxes,projected_lidar_points=points)
    assert len(selected)==2
    batch = collate_fusion_batch(selected)
    verification = dict(state='data_batch_verified',counts=counts,training_samples=[x['metadata']['sample_token'] for x in selected],
                        modalities=['camera','lidar'],radar_available=False,
                        note='CRC-verified public KITTI subset; validation is read for pipeline checks only.')
    (output/'batch_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2),flush=True)
    # Model construction follows successful real-data batch verification.
    torch.set_num_threads(8)
    seed_everything(42)
    config = dict(model_variant='paper_v2',num_classes=1,modality_dropout=0.,dropout_mode='single',fusion_mode='entropy',metric_suppression=False)
    assert torch.cuda.is_available()
    inputs = move_inputs(batch,'cuda')
    model = build_model(config,pretrained=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.0005)
    rows,gradients = [],{}
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        assert prediction['logits'].shape[-1]==2 and not prediction['available'][:,2].any()
        losses = detection_loss(prediction,batch['targets'],batch['metadata'],inputs['camera_mask'])
        assert torch.isfinite(losses['total']) and losses['positive_anchors']>0
        losses['total'].backward()
        if step==0:
            for name in ('camera','lidar','exchange','head'):
                gradients[name] = sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None)
                assert gradients[name]>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        optimizer.step()
        rows.append({key:float(losses[key].detach()) for key in ('total','center_error_meters')})
    mean = lambda key,part:sum(r[key] for r in part)/len(part)
    report = dict(state='completed',steps=80,first_loss=mean('total',rows[:5]),last_loss=mean('total',rows[-5:]),
                  first_center_error_m=mean('center_error_meters',rows[:5]),last_center_error_m=mean('center_error_meters',rows[-5:]),
                  gradients=gradients,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  note='Two repeated training frames, FP32, dropout off, ImageNet camera initialization. Mechanics check; no held-out AP.')
    report['learning_check_passed'] = report['last_loss']<.7*report['first_loss'] and report['last_center_error_m']<report['first_center_error_m']
    (output/'learning_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    assert report['learning_check_passed'],report


if __name__ == '__main__': main()
