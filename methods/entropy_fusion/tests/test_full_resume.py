import json
import sys

import pytest
import torch
from scripts import run_full_experiment as runner


@pytest.mark.parametrize('minimum_map',[0.,.2])
@pytest.mark.parametrize('dataset',['nuscenes','kitti'])
def test_resume_preserves_history_and_rolls_back_metrics(tmp_path, monkeypatch, minimum_map, dataset):
    output = tmp_path/'run'
    training = output/'training'
    training.mkdir(parents=True)
    root = tmp_path/'data'
    checkpoint = dict(config=dict(version='v1.0-trainval' if dataset=='nuscenes' else None, root=str(root),dataset=dataset),
                      step=2, epoch=0, batch_cursor=2)
    torch.save(checkpoint, training/'last.pt')
    (output/'status.json').write_text(json.dumps(dict(pid=123, epochs=1,
        state='failed', error='old error', validation=[], elapsed_seconds=12)))
    original = ''.join(json.dumps(dict(step=i))+'\n' for i in (1, 2, 3))
    (training/'metrics.jsonl').write_text(original)
    log = output/'train_to_epoch_01.log'
    log.write_text('original failure\n')
    monkeypatch.setattr(sys, 'argv', ['runner', '--dataset',dataset,'--resume', '--output', str(output),
        '--root', str(root), '--epochs', '1', '--minimum-first-map',str(minimum_map)])
    def dead_pid(*args):
        raise ProcessLookupError
    monkeypatch.setattr(runner.os, 'kill', dead_pid)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: 'commit\n')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if 'eval.py' in command or 'eval_kitti.py' in command:
            directory = output/'eval_epoch_01'
            directory.mkdir()
            if dataset=='nuscenes':
                (directory/'metrics_summary.json').write_text(json.dumps(dict(mean_ap=.1, nd_score=.2)))
            else:
                (directory/'metrics.json').write_text(json.dumps(dict(car_AP_R40_percent={'3d':{'moderate':10.}})))
    monkeypatch.setattr(runner.subprocess, 'run', run)
    if minimum_map:
        with pytest.raises(RuntimeError, match='below configured floor'):
            runner.main()
    else:
        runner.main()
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == ('failed' if minimum_map else 'completed')
    assert status['resumes'][0]['step'] == 2
    assert status['elapsed_seconds'] >= 12
    assert ('error' in status) == bool(minimum_map)
    assert status['validation'][0]['epoch'] == 1
    if dataset=='kitti':
        assert status['validation'][0]['car_3d_AP_R40_moderate']==.1
        assert 'mAP' not in status['validation'][0] and 'NDS' not in status['validation'][0]
    assert '--resume' in commands[0]
    assert log.read_text().startswith('original failure\n')
    assert [json.loads(row)['step'] for row in (training/'metrics.jsonl').read_text().splitlines()] == [1, 2]
    assert next(training.glob('metrics_before_resume_*.jsonl')).read_text() == original
    assert next(output.glob('status_before_resume_*.json')).exists()
    assert (output/'best.pt').exists()
