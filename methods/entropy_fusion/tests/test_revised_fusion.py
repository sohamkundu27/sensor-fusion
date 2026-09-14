import torch

from models.revised import MeasurementEntropy, ProgressiveFeatureExchange, drop_independent_modalities


def test_sensor_entropy_preserves_channels_and_zero_filled_measurements():
    entropy = MeasurementEntropy()
    ramp = torch.arange(256).reshape(1, 1, 16, 16).float() / 255
    value = torch.cat((ramp, ramp * 0 + .5, ramp * 0), dim=1)
    full = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    expected = torch.tensor([1., 0., 0., 1.]).reshape(1, 4, 1, 1)
    assert torch.allclose(entropy(value, full), expected, atol=1e-6)
    half = full.clone()
    half[..., :8, :] = False
    # Constant measured half and zero-filled missing half yield one bit of entropy.
    result = entropy(value * 0 + .5, half)
    assert torch.allclose(result[:, :3], torch.full((1, 3, 1, 1), 1 / 8))
    assert result[:, 3].item() == .5
    assert not entropy(value, ~full).any()


def test_conditional_independent_dropout_keeps_a_sensor_without_bias_leaks():
    torch.manual_seed(42)
    inputs = [torch.ones(14000, 3, 1, 1) for _ in range(3)]
    masks = [torch.ones(14000, 1, 1, 1, dtype=torch.bool) for _ in range(3)]
    dropped, dropped_masks, available = drop_independent_modalities(inputs, masks, .5, True)
    assert available.any(1).all()
    # At p=.5 the seven nonempty subsets are equiprobable: 3 singletons, 3 pairs,
    # and one all-sensor outcome. This rejects the previous exactly-one-drop rule.
    counts = torch.bincount(available.sum(1), minlength=4).float() / len(available)
    assert torch.allclose(counts[1:], torch.tensor([3 / 7, 3 / 7, 1 / 7]), atol=.02)
    for i in range(3):
        assert not dropped[i][~available[:, i]].any()
        assert not dropped_masks[i][~available[:, i]].any()
    masks[1].zero_()
    assert not drop_independent_modalities(inputs, masks, 1, True)[2][:, 1].any()
    assert (drop_independent_modalities(inputs, masks, 1, True)[2].sum(1) == 1).all()
    assert torch.equal(drop_independent_modalities(inputs, masks, 1, False)[2],
                       torch.tensor([True, False, True]).expand(14000, -1))


def test_progressive_exchange_cross_sensor_feedback_and_absence_mask():
    torch.manual_seed(3)
    block = ProgressiveFeatureExchange((8, 8, 8), out_channels=16)
    features = [torch.rand(1, 8, 4, 4, requires_grad=True) for _ in range(3)]
    entropies = [torch.rand(1, 4, 2, 2) for _ in range(3)]
    available = torch.tensor([[True, True, False]])
    updated, fused = block(features, entropies, available)
    # Only camera feedback is used; its gradient must reach the LiDAR stream.
    updated[0].square().mean().backward()
    assert features[1].grad.abs().sum() > 0
    assert not features[2].grad.any()
    assert block.gate.weight.grad.abs().sum() > 0
    assert not updated[2].any()
    changed = [features[0], features[1], features[2] + 1000]
    altered_entropy = [entropies[0], entropies[1], entropies[2] + 1000]
    updated2, fused2 = block(changed, altered_entropy, available)
    assert torch.equal(fused, fused2)
    assert torch.equal(updated[0], updated2[0])


def test_concat_ablation_has_progressive_exchange_without_entropy_conditioning():
    block = ProgressiveFeatureExchange((8, 8, 8), out_channels=16, fusion_mode='concat')
    features = [torch.randn(1, 8, 4, 4, requires_grad=True) for _ in range(3)]
    updated, fused = block(features, None, torch.ones(1, 3, dtype=torch.bool))
    assert block.gate is None and fused.shape == (1, 16, 4, 4)
    updated[0].square().mean().backward()
    assert features[1].grad.abs().sum() > 0
