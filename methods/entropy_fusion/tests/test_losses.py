import torch
import numpy as np
from models.losses import match_anchors, detection_loss


def test_matching_covers_colliding_ground_truth_and_empty_scene():
    anchors = torch.tensor([[0., 0, 10, 10], [1., 1, 9, 9], [20., 20, 30, 30]])
    boxes = torch.tensor([[0., 0, 10, 10], [0., 0, 10, 10]])
    a, p, n = match_anchors(anchors, boxes, torch.tensor([True, True, False]))
    assert set(a[p].tolist()) == {0, 1}
    assert not (p & n).any() and not n[2]
    _, p, n = match_anchors(anchors, boxes[:0], torch.tensor([True, True, False]))
    assert not p.any() and n.sum() == 2


def test_empty_scene_has_finite_background_gradient():
    pred = {k: torch.zeros(1, 3, n, requires_grad=True) for k,n in
            [('logits', 11), ('boxes', 4), ('boxes3d', 10), ('attributes', 8)]}
    pred['anchors'] = torch.tensor([[0., 0, 10, 10], [1., 1, 9, 9], [20., 20, 30, 30]])
    target = dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long), annotations_3d=[])
    meta = dict(global_to_camera=np.eye(4), ego_to_global=np.eye(4), intrinsic=np.eye(3))
    loss = detection_loss(pred, [target], [meta], torch.ones(1, 1, 32, 32, dtype=torch.bool))
    loss['total'].backward()
    assert torch.isfinite(loss['total']) and pred['logits'].grad.abs().sum() > 0
    assert not pred['boxes3d'].grad.any()
