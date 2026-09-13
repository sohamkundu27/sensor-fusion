import torch
from models.entropy import PatchEntropy
from models.fusion import drop_modalities, FeatureExchange


def test_entropy_constant_uniform_and_missing():
    entropy = PatchEntropy(bins=16)
    x = torch.arange(256).reshape(1, 1, 16, 16).float() / 256
    full = torch.ones_like(x, dtype=torch.bool)
    assert torch.allclose(entropy(x, full), torch.ones(1, 2, 1, 1))
    assert torch.equal(entropy(x * 0, full), torch.tensor([0., 1.]).reshape(1, 2, 1, 1))
    assert not entropy(x, ~full).any()


def test_dropout_gates_prevent_bias_leak_and_disable_at_eval():
    xs = [torch.ones(12, c, 16, 16) for c in (3, 1, 1)]
    masks = [torch.ones(12, 1, 16, 16, dtype=torch.bool) for _ in xs]
    dropped, _, available = drop_modalities(xs, masks, 1, True)
    assert (available.sum(1) == 2).all()
    for i, x in enumerate(dropped):
        assert not x[~available[:, i]].any()
    assert drop_modalities(xs, masks, 1, False)[2].all()
    fusion = FeatureExchange((8, 8, 8), 8)
    features = [torch.rand(1, 8, 2, 2) for _ in range(3)]
    ent = [torch.ones(1, 2, 1, 1) for _ in range(3)]
    active = torch.tensor([[True, False, True]])
    a = fusion(features, ent, active)
    features[1] = features[1] + 100
    assert torch.equal(a, fusion(features, ent, active))
