"""Export local KITTI car predictions and run the pinned official C++ evaluator."""
import argparse
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
import torch
from torch.utils.data import DataLoader
from data import collate_fusion_batch
from data.kitti_dataset import KittiFusionDataset, LOCAL_TO_RECT
from models.factory import build_model
from models.predictions import decode_predictions
from scripts.kitti_evaluator import evaluate_files
from train import move_inputs


def kitti_detection_line(record, metadata):
    center = LOCAL_TO_RECT @ np.asarray(record['translation'])
    width,length,height = record['size']
    bottom = center+np.array([0,height/2,0])
    axis = LOCAL_TO_RECT @ Quaternion(record['rotation']).rotation_matrix[:,0]
    ry = np.arctan2(-axis[2],axis[0])
    alpha = (ry-np.arctan2(bottom[0],bottom[2])+np.pi)%(2*np.pi)-np.pi
    box = np.asarray(record['box2d'],dtype=float).copy()
    affine = metadata['pixel_transform']
    h,w = metadata['original_hw']
    box[[0,2]] = np.clip((box[[0,2]]-affine[0,2])/affine[0,0],0,w)
    box[[1,3]] = np.clip((box[[1,3]]-affine[1,2])/affine[1,1],0,h)
    if box[2]<=box[0] or box[3]<=box[1]: return None
    values = [-1,-1,alpha,*box,height,width,length,*bottom,ry,record['detection_score']]
    if not np.isfinite(values).all(): raise ValueError('Nonfinite KITTI prediction')
    return 'Car '+' '.join(f'{x:.8f}' for x in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--score-threshold',type=float,default=.001)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh evaluation output')
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config = checkpoint['config']
    if config.get('dataset')!='kitti' or config.get('num_classes')!=1: raise ValueError('Expected KITTI car checkpoint')
    root = args.root or Path(config['root']).expanduser()
    dataset = KittiFusionDataset(root,'val',image_hw=config['image_hw'])
    loader = DataLoader(dataset,batch_size=2,num_workers=2,multiprocessing_context='spawn',
                        timeout=120,collate_fn=collate_fusion_batch,pin_memory=True)
    model = build_model(config).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    detections = args.output/'detections'
    detections.mkdir(parents=True)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = move_inputs(batch,'cuda')
            prediction = model(inputs)  # FP32 evaluation, as in nuScenes evaluation.
            records = decode_predictions(prediction,batch['metadata'],args.score_threshold,include_boxes2d=True)
            for rows,meta in zip(records,batch['metadata']):
                lines = [line for row in rows if (line:=kitti_detection_line(row,meta)) is not None]
                (detections/f"{meta['sample_token']}.txt").write_text('\n'.join(lines)+ ('\n' if lines else ''))
            count += len(records)
            if count%100==0: print(json.dumps(dict(evaluated=count,total=len(dataset))),flush=True)
    report = evaluate_files(root/'training/label_2',detections,dataset.ids,args.output/'official')
    report.update(checkpoint=str(args.checkpoint),score_threshold=args.score_threshold,frames=len(dataset),
                  note='Local held-out labeled split; not a KITTI test-server submission. Tiny subsets are mechanics checks only.')
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
