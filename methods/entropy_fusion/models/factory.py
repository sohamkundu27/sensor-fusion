"""Explicit model revisions keep completed baseline checkpoints reproducible."""
from .detector import EntropyFusionDetector


def build_model(config, pretrained=False):
    variant = config.get('model_variant', 'baseline')
    if variant == 'baseline':
        return EntropyFusionDetector(pretrained=pretrained, modality_dropout=config['modality_dropout'])
    if variant == 'paper_v2':
        from .revised import ReimplementedEntropyFusionDetector
        return ReimplementedEntropyFusionDetector(pretrained=pretrained,
            modality_dropout=config['modality_dropout'], fusion_mode=config.get('fusion_mode','entropy'))
    raise ValueError(f'Unknown model variant: {variant}')
