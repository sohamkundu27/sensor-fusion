"""Physical error scales and detached confidence supervision for the revised head."""
import math

import numpy as np
import torch

from models.boxes import encode_3d, prepare_targets
from models.losses import detection_loss


def example(revised=True, velocity_valid=True, empty=False):
    anchors = torch.tensor([[12., 12., 20., 20.], [32., 0., 40., 8.],
                            [48., 0., 56., 8.], [64., 0., 72., 8.],
                            [32., 32., 40., 40.], [48., 32., 56., 40.],
                            [64., 32., 72., 40.]])
    meta = dict(global_to_camera=np.eye(4), ego_to_global=np.eye(4),
                intrinsic=np.array([[20., 0., 16.], [0., 20., 16.], [0., 0., 1.]]))
    annotation = dict(translation=[0., 0., 20.], size=[2., 4., 1.5],
                      rotation=[1., 0., 0., 0.], num_lidar_pts=5, num_radar_pts=0,
                      velocity_global=[0., 0., 0.] if velocity_valid else [float('nan')]*3,
                      velocity_valid=velocity_valid, attribute_names=['vehicle.moving'])
    target = dict(boxes=anchors[:1].clone(), labels=torch.tensor([1]),
                  annotations_3d=[annotation])
    prediction = {key: torch.zeros(1, len(anchors), width) for key, width in
                  [('logits', 11), ('boxes', 4), ('boxes3d', 10), ('attributes', 8), ('quality', 1)]}
    prediction['anchors'] = anchors
    prediction['revised'] = revised
    prepared = prepare_targets(target, meta, 'cpu')
    prediction['boxes3d'][0, 0] = encode_3d(prepared['values'], anchors[:1])[0]
    for key in ('logits', 'boxes', 'boxes3d', 'attributes', 'quality'):
        prediction[key].requires_grad_()
    if empty:
        target = dict(boxes=anchors[:0], labels=torch.empty(0, dtype=torch.long), annotations_3d=[])
    return prediction, [target], [meta], torch.ones(1, 1, 64, 80, dtype=torch.bool)


def test_baseline_loss_preserves_original_values_and_keys():
    args = example(revised=False)
    losses = detection_loss(*args)
    # One exact positive, three background hard negatives, exact regression.
    torch.testing.assert_close(losses['logits'], torch.tensor(4*math.log(11)))
    torch.testing.assert_close(losses['boxes3d'], torch.tensor(0.))
    torch.testing.assert_close(losses['total'], torch.tensor(4*math.log(11)+.2*math.log(8)))
    assert set(losses) == {'logits', 'boxes', 'boxes3d', 'attributes', 'total', 'positive_anchors'}
    args[0].pop('revised')
    args[0].pop('quality')
    old_interface = detection_loss(*args)
    for key in losses:
        if torch.is_tensor(losses[key]):
            torch.testing.assert_close(losses[key], old_interface[key], rtol=0, atol=0)
        else:
            assert losses[key] == old_interface[key]


def test_two_meter_depth_error_has_meaningful_metric_penalty():
    args = example()
    with torch.no_grad():
        args[0]['boxes3d'][0, 0, 2] = math.log(22.)
    losses = detection_loss(*args)
    torch.testing.assert_close(losses['center_meters'], torch.tensor(1.5), atol=1e-5, rtol=0)
    torch.testing.assert_close(losses['center_error_meters'], torch.tensor(2.), atol=1e-5, rtol=0)
    assert .25*losses['center_meters'] > 50*losses['boxes3d']
    assert not losses['center_error_meters'].requires_grad
    losses['center_meters'].backward()
    assert args[0]['boxes3d'].grad[0, 0, 2] > 0


def test_velocity_uses_meters_per_second_and_masks_unknown_values():
    known = example()
    with torch.no_grad():
        known[0]['boxes3d'][0, 0, 8] = .1  # One m/s along a camera-relative ground axis.
    losses = detection_loss(*known)
    torch.testing.assert_close(losses['velocity_mps'], torch.tensor(.5))
    torch.testing.assert_close(losses['boxes3d'], torch.tensor(0.))
    losses['velocity_mps'].backward()
    assert known[0]['boxes3d'].grad[0, 0, 8] > 0
    unknown = example(velocity_valid=False)
    with torch.no_grad():
        unknown[0]['boxes3d'][0, 0, 8:10] = torch.tensor([4., -8.])
    masked = detection_loss(*unknown)
    torch.testing.assert_close(masked['velocity_mps'], torch.tensor(0.))
    masked['total'].backward()
    assert not unknown[0]['boxes3d'].grad[..., 8:10].any()


def test_quality_uses_five_hard_negatives_and_detaches_metric_target():
    args = example()
    with torch.no_grad():
        args[0]['boxes3d'][0, 0, 2] = math.log(22.)
    losses = detection_loss(*args)
    torch.testing.assert_close(losses['logits'], torch.tensor(6*math.log(11)))
    losses['quality'].backward()
    gradient = args[0]['quality'].grad[0, :, 0]
    torch.testing.assert_close(gradient[0], torch.tensor(.5-math.exp(-1)), atol=1e-6, rtol=0)
    assert (gradient[1:] > 0).sum() == 5
    assert torch.isfinite(gradient).all()
    assert args[0]['boxes3d'].grad is None or not args[0]['boxes3d'].grad.any()


def test_revised_empty_scene_and_extreme_depth_have_finite_gradients():
    for empty in (True, False):
        args = example(empty=empty)
        with torch.no_grad():
            args[0]['boxes3d'][0, 0, 2] = 1000.
        losses = detection_loss(*args)
        losses['total'].backward()
        assert torch.isfinite(losses['total'])
        for key in ('logits', 'boxes', 'boxes3d', 'attributes', 'quality'):
            assert torch.isfinite(args[0][key].grad).all()
        if empty:
            assert losses['center_error_meters'] == 0
            assert not args[0]['boxes3d'].grad.any()
