"""Prepare a matched nuScenes gating pair; training requires explicit --run.

Both arms retain entropy concatenation. Only multiplicative gating is ablated.
Preparation checks identical initialization and freezes source/config hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
CONFIGS = [BASE/f'configs/nuscenes_gating_{arm}.json' for arm in ('on','off')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint():
    files = [BASE/'train.py', BASE/'eval.py', BASE/'scripts/run_full_experiment.py', Path(__file__).resolve(), *CONFIGS]
    for directory in ('models','data'):
        files.extend(sorted((BASE/directory).glob('*.py')))
    return {str(p.relative_to(BASE)):digest(p) for p in files}


def metadata_fingerprint(config):
    directory=Path(config['root'])/config['version']
    files=sorted(directory.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No dataset metadata in {directory}')
    return {p.name:digest(p) for p in files}


def configs():
    pair = [json.loads(p.read_text()) for p in CONFIGS]
    different = [k for k in pair[0].keys() | pair[1].keys() if pair[0].get(k)!=pair[1].get(k)]
    if different != ['entropy_gating'] or not pair[0]['entropy_gating'] or pair[1]['entropy_gating']:
        raise ValueError(f'Expected ONLY entropy_gating true/false difference; found {different}')
    if any(c['model_variant']!='paper_v2' or c['fusion_mode']!='entropy' or c['version']!='v1.0-trainval' or c['split']!='train' for c in pair):
        raise ValueError('Expected full nuScenes paper_v2 with entropy conditioning in both arms')
    return pair


def commands(output,pair,resume=False):
    result=[]
    for arm,path,c in zip(('on','off'),CONFIGS,pair):
        cmd=[sys.executable,'-u',str(BASE/'scripts/run_full_experiment.py'),'--dataset','nuscenes',
             '--config',str(path),'--root',c['root'],'--epochs',str(c['epochs']),
             '--validate-every','5','--minimum-first-map','0','--output',str(output/arm)]
        if resume: cmd.append('--resume-if-present')
        result.append(cmd)
    return result


def prepare(output,pair):
    import torch
    from models.factory import build_model
    from train import seed_everything
    torch.set_num_threads(8)
    weights=[];states=[];counts=[]
    for c in pair:
        seed_everything(c['seed'])
        model=build_model(c,pretrained=True)
        h=hashlib.sha256()
        for name,value in model.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
        weights.append(h.hexdigest());counts.append(sum(p.numel() for p in model.parameters()))
        states.append({k:tuple(v.shape) for k,v in model.state_dict().items()})
        del model
    if weights[0]!=weights[1] or states[0]!=states[1] or counts[0]!=counts[1]:
        raise AssertionError('Initial model parameters/buffers/shapes differ')
    from data import NuScenesFusionDataset
    c=pair[0]
    dataset=NuScenesFusionDataset(c['root'],version=c['version'],split=c['split'],cameras=c['cameras'],image_hw=c['image_hw'],sensor_encoding=c['sensor_encoding'])
    item_count=len(dataset)
    items_hash=hashlib.sha256(json.dumps(dataset.items).encode()).hexdigest()
    # Same dedicated generator as train.py. Ordering proof covers every epoch.
    order_hashes=[]
    from torch.utils.data import DataLoader
    for epoch in range(c['epochs']):
        loader=DataLoader(range(len(dataset)),batch_size=c['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(c['seed']+epoch))
        h=hashlib.sha256()
        for ids in loader:h.update(ids.numpy().tobytes())
        order_hashes.append(h.hexdigest())
    del dataset
    manifest=dict(state='prepared_not_started',comparison='multiplicative_gate_only; entropy and coverage concatenation retained in BOTH arms',
                  seed=c['seed'],source_fingerprint=source_fingerprint(),metadata_sha256=metadata_fingerprint(c),initial_state_sha256=weights,
                  parameters=counts,train_views=item_count,ordered_train_items_sha256=items_hash,
                  epoch_shuffle_sha256=order_hashes,commands=commands(output,pair),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
                  python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                  evaluation=dict(split='val',epochs=[1,5,10,15,20],primary='epoch20 mAP and NDS',secondary='best validation mAP with corresponding NDS',score_threshold=.001,topk_per_view=100,metric_suppression=False,quality_scoring=True),
                  limitations=['CUDA kernels are not forced deterministic; seed does not guarantee bitwise-identical training.',
                               'AMP overflow and optimizer skips may diverge due to treatment; report counts in each arm.',
                               'Gating-off gate parameters remain registered but receive no gradient; effective used capacity differs by the intended treatment.',
                               'One seed is a paired exploratory comparison, not a variance estimate.',
                               'Source/config, all dataset JSON metadata and train-item ordering are hashed; raw sensor files are not fully rehashed.',
                               'Run timing/thermals and OS scheduling cannot be held identical.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'outputs/nuscenes_gating_pair_seed42')
    parser.add_argument('--run',action='store_true',help='Start both prepared 20-epoch runs sequentially')
    parser.add_argument('--resume',action='store_true',help='Resume interrupted arms when used with --run')
    args=parser.parse_args();output=args.output.resolve();pair=configs()
    if args.resume and not args.run:parser.error('--resume requires --run')
    if not args.run:
        if any((output/arm/'status.json').exists() for arm in ('on','off')):
            raise FileExistsError('Do not overwrite provenance after a run starts')
        prepare(output,pair);return
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['source_fingerprint']!=source_fingerprint():
        raise RuntimeError('Source/config changed after preparation; prepare a fresh pair before training')
    import torch
    expected_environment=(output/'environment.txt').read_text()
    for cmd in commands(output,pair,args.resume):
        if manifest['source_fingerprint']!=source_fingerprint():
            raise RuntimeError('Source/config changed between arms; refusing a confounded comparison')
        if (manifest['torch']!=torch.__version__ or manifest['cuda']!=torch.version.cuda or manifest['python']!=sys.version
                or expected_environment!=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)):
            raise RuntimeError('Python/CUDA/package environment changed after preparation')
        if manifest['metadata_sha256']!=metadata_fingerprint(pair[0]):
            raise RuntimeError('Dataset metadata changed after preparation; refusing changed sample order/targets/calibration')
        manifest.update(state='running',active_output=cmd[cmd.index('--output')+1])
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        try:
            subprocess.run(cmd,cwd=BASE,check=True)
        except BaseException as error:
            manifest.update(state='interrupted_or_failed',error=f'{type(error).__name__}: {error}')
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            raise
    manifest.update(state='completed')
    manifest.pop('error',None)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
