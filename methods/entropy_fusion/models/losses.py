"""SSD matching, hard-negative CE and masked 2D/metric-3D regression."""
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou
from .boxes import prepare_targets, encode_boxes, encode_3d


@torch.no_grad()
def match_anchors(anchors, boxes, valid, positive_iou=.5, negative_iou=.4):
    assignment = torch.full((len(anchors),), -1, dtype=torch.long, device=anchors.device)
    positive = torch.zeros(len(anchors), dtype=torch.bool, device=anchors.device)
    negative = valid.clone()
    if len(boxes) == 0:
        return assignment, positive, negative
    iou = box_iou(anchors, boxes)
    best, assignment = iou.max(1)
    positive = (best >= positive_iou) & valid
    negative = (best < negative_iou) & valid
    # Unique forced matches: colliding GT best anchors must not overwrite each other.
    candidates = iou.masked_fill(~valid[:, None], -1)
    for _ in range(min(len(boxes), int(valid.sum()))):
        flat = candidates.argmax()
        anchor, gt = flat // len(boxes), flat % len(boxes)
        if candidates[anchor, gt] < 0:
            break
        assignment[anchor] = gt
        positive[anchor], negative[anchor] = True, False
        candidates[anchor, :] = -1
        candidates[:, gt] = -1
    return assignment, positive, negative


def detection_loss(prediction, targets, metadata, camera_masks):
    # Matching and regression in FP32 even under autocast.
    anchors = prediction['anchors'].float()
    ac = (anchors[:, :2] + anchors[:, 2:]) / 2
    losses = {key: prediction[key].float().sum()*0 for key in ('logits', 'boxes', 'boxes3d', 'attributes')}
    total_positive = 0
    h, w = camera_masks.shape[-2:]
    for i, (raw, meta) in enumerate(zip(targets, metadata)):
        target = prepare_targets(raw, meta, anchors.device)
        valid = camera_masks[i, 0, ac[:, 1].long().clamp(0, h-1), ac[:, 0].long().clamp(0, w-1)].bool()
        assignment, positive, negative = match_anchors(anchors, target['boxes'], valid)
        count = int(positive.sum())
        total_positive += count
        labels = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
        if count:
            labels[positive] = target['labels'][assignment[positive]]
        ce = F.cross_entropy(prediction['logits'][i].float(), labels, reduction='none')
        num_negative = min(int(negative.sum()), max(3*count, 100 if count == 0 else 0))
        negative_ce = ce[negative].topk(num_negative).values if num_negative else ce[:0]
        # Empty scenes still teach background without an unbounded loss scale.
        normalizer = max(count, num_negative if count == 0 else 1)
        losses['logits'] = losses['logits'] + (ce[positive].sum()+negative_ce.sum())/normalizer
        if count:
            idx = assignment[positive]
            losses['boxes'] = losses['boxes'] + F.smooth_l1_loss(prediction['boxes'][i, positive].float(),
                encode_boxes(target['boxes'][idx], anchors[positive]), reduction='sum') / count
            truth3d = encode_3d(target['values'][idx], anchors[positive])
            error = F.smooth_l1_loss(prediction['boxes3d'][i, positive].float(), truth3d, reduction='none')
            weight = torch.ones_like(error)
            weight[:, 8:10] = target['velocity_valid'][idx, None]
            losses['boxes3d'] = losses['boxes3d'] + (error * weight).sum()/count
            attrs = target['attributes'][idx]
            attr_valid = attrs >= 0
            if attr_valid.any():
                losses['attributes'] = losses['attributes'] + F.cross_entropy(
                    prediction['attributes'][i, positive][attr_valid].float(), attrs[attr_valid])
    losses = {key: value/len(targets) for key, value in losses.items()}
    losses['total'] = losses['logits'] + losses['boxes'] + losses['boxes3d'] + .2*losses['attributes']
    losses['positive_anchors'] = total_positive
    return losses
