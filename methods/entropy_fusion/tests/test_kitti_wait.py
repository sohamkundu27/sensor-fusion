import json
import sys
import pytest
from scripts import run_kitti_when_ready as runner


@pytest.mark.parametrize('download_state',['verified','failed'])
def test_queue_starts_only_after_verified_data(tmp_path,monkeypatch,download_state):
    root = tmp_path/'data'
    root.mkdir()
    (root/'subset_manifest.json').write_text(json.dumps(dict(state=download_state)))
    output = tmp_path/'experiment'
    monkeypatch.setattr(sys,'argv',['wait','--root',str(root),'--output',str(output)])
    checked,commands = [],[]
    monkeypatch.setattr(runner,'validate_download',lambda *args:checked.append(args))
    monkeypatch.setattr(runner.subprocess,'run',lambda command,**kwargs:commands.append(command))
    if download_state=='failed':
        with pytest.raises(RuntimeError,match='downloader failed'): runner.main()
        assert not checked and not commands
    else:
        runner.main()
        assert len(checked)==1 and len(commands)==1
        assert '--resume-if-present' in commands[0] and commands[0][commands[0].index('--dataset')+1]=='kitti'


def test_ready_check_rejects_subset_and_overlap(tmp_path):
    with pytest.raises(ValueError,match='3712/3769'):
        runner.validate_download(tmp_path,dict(state='verified',train_count=32,val_count=8))
    with pytest.raises(ValueError,match='overlapping'):
        runner.validate_download(tmp_path,dict(state='verified',train_count=3712,val_count=3769,
            subsets=dict(train=['000000']*3712,val=['000000']*3769)))
