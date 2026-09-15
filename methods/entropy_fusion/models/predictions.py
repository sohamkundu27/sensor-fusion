"""Decode view detections and merge them into nuScenes global-coordinate results."""
import numpy as np
import torch
from torchvision.ops import batched_nms
from data import CLASSES
from .boxes import ATTRIBUTES, decode_boxes, decode_3d

ALLOWED_ATTRIBUTES = {
    'car': (5, 6, 7), 'truck': (5, 6, 7), 'bus': (5, 6, 7),
    'trailer': (5, 6, 7), 'construction_vehicle': (5, 6, 7),
    'pedestrian': (2, 3, 4), 'bicycle': (0, 1), 'motorcycle': (0, 1),
    'traffic_cone': (), 'barrier': (),
}
# Conservative same-class ground-plane distance suppression across overlapping views.
MERGE_RADIUS = {'car': 1., 'truck': 2., 'bus': 2., 'trailer': 2., 'construction_vehicle': 1.,
                'pedestrian': .3, 'bicycle': .5, 'motorcycle': .5, 'traffic_cone': .2, 'barrier': .5}


@torch.no_grad()
def metric_nms(code, anchors, scores, labels, metadata, topk):
    """Suppress only nearby same-class 3D centers, keeping depth-separated boxes."""
    if len(code) == 0:
        return torch.empty(0, dtype=torch.long, device=code.device)
    center = (anchors[:,:2]+anchors[:,2:])*.5
    size = (anchors[:,2:]-anchors[:,:2]).clamp_min(1e-4)
    uv = code[:,:2]*size+center
    k = torch.as_tensor(metadata['intrinsic'],device=code.device,dtype=code.dtype)
    c2g = torch.as_tensor(np.linalg.inv(metadata['global_to_camera']),device=code.device,dtype=code.dtype)
    xyz = (torch.cat((uv,torch.ones_like(uv[:,:1])),1)@torch.linalg.inv(k).T)*code[:,2:3].clamp(0,np.log(100)).exp()
    xy = (xyz@c2g[:3,:3].T+c2g[:3,3])[:,:2].cpu().numpy()
    classes = labels.cpu().numpy()
    order = scores.argsort(descending=True).cpu().numpy()
    kept = []
    while len(order) and len(kept) < topk:
        index = int(order[0])
        kept.append(index)
        remaining = order[1:]
        distance = np.linalg.norm(xy[remaining]-xy[index],axis=1)
        suppress = (classes[remaining]==classes[index]) & (distance<MERGE_RADIUS[CLASSES[int(classes[index])]])
        order = remaining[~suppress]
    return torch.as_tensor(kept,device=code.device,dtype=torch.long)


@torch.no_grad()
def decode_predictions(prediction, metadata, score_threshold=.05, topk=100, include_boxes2d=False):
    output = []
    for i, meta in enumerate(metadata):
        scores, labels = prediction['logits'][i].float().softmax(-1)[:, 1:].max(-1)
        if prediction.get('revised', False) and prediction.get('quality_scoring', True):
            scores = scores * prediction['quality'][i,:,0].float().sigmoid()
        valid = prediction['valid_anchors'][i] if 'valid_anchors' in prediction else torch.ones_like(scores, dtype=torch.bool)
        candidate = torch.where((scores >= score_threshold) & valid)[0]
        candidate = candidate[scores[candidate].argsort(descending=True)[:1000]]
        boxes = decode_boxes(prediction['boxes'][i, candidate].float(), prediction['anchors'][candidate])
        if prediction.get('revised', False) and prediction.get('metric_suppression', True):
            keep = metric_nms(prediction['boxes3d'][i,candidate].float(), prediction['anchors'][candidate],
                              scores[candidate], labels[candidate], meta, topk)
        else:
            keep = batched_nms(boxes, scores[candidate], labels[candidate], .5)[:topk]
        indices = candidate[keep]
        xyz, sizes, rotations, velocity = decode_3d(prediction['boxes3d'][i, indices].float(), prediction['anchors'][indices], meta)
        records = []
        for j, index in enumerate(indices.tolist()):
            name = CLASSES[int(labels[index])]
            allowed = ALLOWED_ATTRIBUTES[name]
            attribute = ATTRIBUTES[allowed[int(prediction['attributes'][i, index, list(allowed)].argmax())]] if allowed else ''
            values = [*xyz[j].tolist(), *sizes[j].tolist(), *velocity[j].tolist(), *rotations[j]]
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite decoded 3D detection')
            records.append(dict(sample_token=meta['sample_token'], translation=xyz[j].tolist(),
                                size=sizes[j].tolist(), rotation=rotations[j], velocity=velocity[j].tolist(),
                                detection_name=name, detection_score=float(scores[index]), attribute_name=attribute))
            if include_boxes2d:
                records[-1]['box2d'] = boxes[keep[j]].tolist()
        output.append(records)
    return output


def merge_views(records, max_boxes=500):
    kept = []
    for record in sorted(records, key=lambda r: r['detection_score'], reverse=True):
        name = record['detection_name']
        xy = np.asarray(record['translation'][:2])
        if any(k['detection_name'] == name and np.linalg.norm(xy-np.asarray(k['translation'][:2])) < MERGE_RADIUS[name]
               for k in kept):
            continue
        kept.append(record)
        if len(kept) == max_boxes:
            break
    return kept
