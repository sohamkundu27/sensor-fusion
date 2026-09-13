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
def decode_predictions(prediction, metadata, score_threshold=.05, topk=100):
    output = []
    for i, meta in enumerate(metadata):
        scores, labels = prediction['logits'][i].float().softmax(-1)[:, 1:].max(-1)
        valid = prediction['valid_anchors'][i] if 'valid_anchors' in prediction else torch.ones_like(scores, dtype=torch.bool)
        candidate = torch.where((scores >= score_threshold) & valid)[0]
        candidate = candidate[scores[candidate].argsort(descending=True)[:1000]]
        boxes = decode_boxes(prediction['boxes'][i, candidate].float(), prediction['anchors'][candidate])
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
