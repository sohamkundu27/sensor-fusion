"""The gate-only control preserves architecture, entropy inputs, and resume semantics."""
import json
from pathlib import Path
import sys

import pytest
import torch

from models.factory import build_model
from models.revised import ProgressiveFeatureExchange
import train


def test_gate_switch_preserves_all_parameters_initialization_and_rng():
    config = dict(model_variant='paper_v2', modality_dropout=.5, fusion_mode='entropy')
    torch.manual_seed(42)
    enabled = build_model(config)
    after_enabled = torch.get_rng_state()
    torch.manual_seed(42)
    disabled = build_model(dict(config, entropy_gating=False))
    assert torch.equal(after_enabled, torch.get_rng_state())
    assert enabled.entropy_gating and not disabled.entropy_gating
    assert sum(p.numel() for p in enabled.parameters()) == sum(p.numel() for p in disabled.parameters())
    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    for name, value in enabled.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[name], rtol=0, atol=0)
    assert all(block.gate is not None and not block.entropy_gating for block in disabled.exchange)
    # Old checkpoints have no additional switch buffer/parameter to deserialize.
    disabled.load_state_dict(enabled.state_dict(), strict=True)


@pytest.mark.parametrize('enabled', [True, False])
def test_only_feature_weights_change_and_missing_streams_stay_masked(enabled):
    torch.manual_seed(7)
    block = ProgressiveFeatureExchange((8, 8, 8), out_channels=16, entropy_gating=enabled)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.bias.zero_()  # Known .5 weight when enabled.
    features = [torch.rand(1, 8, 4, 4, requires_grad=True) for _ in range(3)]
    entropies = [torch.rand(1, 4, 2, 2) for _ in range(3)]
    available = torch.tensor([[True, True, False]])
    captured = []
    hook = block.project.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    updated, fused = block(features, entropies, available)
    weight = .5 if enabled else 1.
    torch.testing.assert_close(captured[0][:, :8], features[0]*weight, rtol=0, atol=0)
    torch.testing.assert_close(captured[0][:, 8:16], features[1]*weight, rtol=0, atol=0)
    assert not captured[0][:, 16:24].any()
    assert captured[0][:, 24:32].abs().sum() > 0  # Entropy concatenation retained in both arms.
    assert not captured[0][:, 32:].any()  # Missing radar entropy remains masked too.
    assert not updated[2].any()
    fused.square().mean().backward()
    assert features[0].grad.abs().sum() > 0 and features[1].grad.abs().sum() > 0
    assert not features[2].grad.any()
    if enabled:
        assert block.gate.weight.grad.abs().sum() > 0
    else:
        assert block.gate.weight.grad is None and block.gate.bias.grad is None
    # Values and entropy from a missing sensor cannot influence surviving streams.
    _, altered = block([features[0], features[1], features[2]+1000],
                       [entropies[0], entropies[1], entropies[2]+1000], available)
    torch.testing.assert_close(fused, altered, rtol=0, atol=0)
    hook.remove()


def test_off_matches_identity_gate_but_preserves_entropy_conditioning():
    torch.manual_seed(9)
    block = ProgressiveFeatureExchange((8, 8, 8), out_channels=16)
    features = [torch.randn(1, 8, 4, 4) for _ in range(3)]
    entropies = [torch.rand(1, 4, 2, 2) for _ in range(3)]
    available = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.bias.fill_(100.)  # sigmoid rounds to exactly one in FP32.
        before_rng = torch.get_rng_state()
        updates_on, fused_on = block(features, entropies, available)
        block.entropy_gating = False
        updates_off, fused_off = block(features, entropies, available)
        assert torch.equal(before_rng, torch.get_rng_state())
        torch.testing.assert_close(fused_on, fused_off, rtol=0, atol=0)
        for on, off in zip(updates_on, updates_off):
            torch.testing.assert_close(on, off, rtol=0, atol=0)
        _, changed_condition = block(features, [e+1 for e in entropies], available)
        assert not torch.allclose(fused_off, changed_condition)


@pytest.mark.parametrize('variant,mode', [('baseline', 'entropy'), ('paper_v2', 'concat')])
def test_rejects_gate_control_for_incompatible_architecture(variant, mode):
    with pytest.raises(ValueError, match='requires'):
        build_model(dict(model_variant=variant, fusion_mode=mode, modality_dropout=.5, entropy_gating=False))


def test_cli_switch_is_explicit_and_defaults_to_legacy_behavior(tmp_path, monkeypatch):
    config = json.loads((Path(train.__file__).parent/'configs/paper_v2.json').read_text())
    config['amp'] = False
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    argv = ['train.py', '--config', str(path), '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', argv)
    assert train.parse_args().entropy_gating is True
    monkeypatch.setattr(sys, 'argv', argv+['--no-entropy-gating'])
    assert train.parse_args().entropy_gating is False
    config['entropy_gating'] = False
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', argv+['--entropy-gating'])
    assert train.parse_args().entropy_gating is True


@pytest.mark.parametrize('saved,current,compatible', [
    (True, False, False), (False, True, False), (None, False, False),
    (None, True, True), (True, True, True), (False, False, True),
])
def test_resume_guard_prevents_switching_arms_and_accepts_old_defaults(tmp_path, monkeypatch, saved, current, compatible):
    config = json.loads((Path(train.__file__).parent/'configs/paper_v2.json').read_text())
    config.update(amp=False, entropy_gating=current)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    checkpoint_path = tmp_path/'checkpoint.pt'
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(path), '--device', 'cpu',
        '--output', str(tmp_path/'run'), '--resume', str(checkpoint_path)])
    args = train.parse_args()
    saved_config = vars(args).copy()
    if saved is None:
        saved_config.pop('entropy_gating')
    else:
        saved_config['entropy_gating'] = saved
    torch.save(dict(format_version=1, config=saved_config), checkpoint_path)
    monkeypatch.setattr(train, 'parse_args', lambda: args)
    class GuardPassed(Exception):
        pass
    def stop_before_data(*args, **kwargs):
        raise GuardPassed
    monkeypatch.setattr(train, 'NuScenesFusionDataset', stop_before_data)
    if compatible:
        with pytest.raises(GuardPassed):
            train.main()
    else:
        with pytest.raises(ValueError, match='Resume setting differs: entropy_gating'):
            train.main()
