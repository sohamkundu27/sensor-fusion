"""SSD matching, hard-negative CE and masked 2D/metric-3D regression."""
import math
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
    revised = bool(prediction.get('revised', False))
    if revised:
        losses.update(center_meters=prediction['boxes3d'].float().sum()*0,
                      velocity_mps=prediction['boxes3d'].float().sum()*0,
                      quality=prediction['quality'].float().sum()*0)
        center_error_sum = anchors.new_zeros(())
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
        negative_ratio = 5 if revised else 3
        num_negative = min(int(negative.sum()), max(negative_ratio*count, 100 if count == 0 else 0))
        negative_indices = torch.where(negative)[0]
        chosen_negative = negative_indices[ce[negative].topk(num_negative).indices] if num_negative else negative_indices[:0]
        negative_ce = ce[chosen_negative]
        # Empty scenes still teach background without an unbounded loss scale.
        normalizer = max(count, num_negative if count == 0 else 1)
        losses['logits'] = losses['logits'] + (ce[positive].sum()+negative_ce.sum())/normalizer
        if revised:
            # The same hard negatives supervise a zero localization-quality target.
            quality_logits = prediction['quality'][i, :, 0].float()
            selected_quality = quality_logits[chosen_negative]
            losses['quality'] = losses['quality'] + F.binary_cross_entropy_with_logits(
                selected_quality, torch.zeros_like(selected_quality), reduction='sum') / normalizer
        if count:
            idx = assignment[positive]
            losses['boxes'] = losses['boxes'] + F.smooth_l1_loss(prediction['boxes'][i, positive].float(),
                encode_boxes(target['boxes'][idx], anchors[positive]), reduction='sum') / count
            truth3d = encode_3d(target['values'][idx], anchors[positive])
            error = F.smooth_l1_loss(prediction['boxes3d'][i, positive].float(), truth3d, reduction='none')
            weight = torch.ones_like(error)
            # Revised velocity is supervised in physical units below, replacing the
            # weak velocity/10 term while keeping the existing UV/log-depth codes.
            weight[:, 8:10] = 0 if revised else target['velocity_valid'][idx, None]
            losses['boxes3d'] = losses['boxes3d'] + (error * weight).sum()/count
            if revised:
                code = prediction['boxes3d'][i, positive].float()
                positive_anchors = anchors[positive]
                anchor_center = (positive_anchors[:, :2] + positive_anchors[:, 2:]) / 2
                anchor_size = (positive_anchors[:, 2:] - positive_anchors[:, :2]).clamp_min(1e-4)
                predicted_uv = code[:, :2] * anchor_size + anchor_center
                values = target['values'][idx]
                inverse_intrinsic = torch.linalg.inv(torch.as_tensor(meta['intrinsic'],
                    device=anchors.device, dtype=torch.float32))
                def camera_center(uv, log_depth):
                    rays = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=1) @ inverse_intrinsic.T
                    # Broad safety bounds protect exponentiation; ordinary 1-100m
                    # examples retain a gradient through the metric depth loss.
                    return rays * log_depth.clamp(-2., math.log(150.)).exp()[:, None]
                predicted_center = camera_center(predicted_uv, code[:, 2])
                true_center = camera_center(values[:, :2], values[:, 2])
                losses['center_meters'] = losses['center_meters'] + F.smooth_l1_loss(
                    predicted_center, true_center, beta=1., reduction='sum') / count
                center_error = torch.linalg.vector_norm(predicted_center.detach()-true_center, dim=1)
                center_error_sum = center_error_sum + center_error.sum()
                # A 2m position error gives quality exp(-1); this target is detached
                # so the quality objective cannot move the regressed box itself.
                quality_target = torch.exp(-center_error / 2.)
                losses['quality'] = losses['quality'] + F.binary_cross_entropy_with_logits(
                    quality_logits[positive], quality_target, reduction='sum') / count
                velocity_valid = target['velocity_valid'][idx]
                if velocity_valid.any():
                    losses['velocity_mps'] = losses['velocity_mps'] + F.smooth_l1_loss(
                        code[velocity_valid, 8:10]*10., values[velocity_valid, 8:10]*10.,
                        beta=1., reduction='sum') / velocity_valid.sum()
            attrs = target['attributes'][idx]
            attr_valid = attrs >= 0
            if attr_valid.any():
                losses['attributes'] = losses['attributes'] + F.cross_entropy(
                    prediction['attributes'][i, positive][attr_valid].float(), attrs[attr_valid])
    losses = {key: value/len(targets) for key, value in losses.items()}
    losses['total'] = losses['logits'] + losses['boxes'] + losses['boxes3d'] + .2*losses['attributes']
    if revised:
        losses['total'] = losses['total'] + float(prediction.get('metric_center_weight',.25))*losses['center_meters'] + .1*losses['velocity_mps'] + .5*losses['quality']
        losses['center_error_meters'] = center_error_sum / max(total_positive, 1)
    losses['positive_anchors'] = total_positive
    return losses
