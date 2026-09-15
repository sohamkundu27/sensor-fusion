"""Check official-evaluator integration against perfect and displaced 3D boxes."""
import json
from pathlib import Path
import tempfile
from scripts.kitti_evaluator import evaluate_files


def main():
    with tempfile.TemporaryDirectory(prefix='kitti-eval-check-') as temporary:
        root = Path(temporary)
        ids = [f'{i:06d}' for i in range(50)]
        for name in ('labels','perfect','displaced'): (root/name).mkdir()
        for sample in ids:
            label = 'Car 0 0 0 20 20 100 100 2 2 4 0 2 20 0'
            (root/'labels'/f'{sample}.txt').write_text(label+'\n')
            (root/'perfect'/f'{sample}.txt').write_text(label+' 0.9\n')
            shifted = 'Car 0 0 0 20 20 100 100 2 2 4 100 2 20 0 0.9\n'
            (root/'displaced'/f'{sample}.txt').write_text(shifted)
        perfect = evaluate_files(root/'labels',root/'perfect',ids,root/'perfect_eval')
        displaced = evaluate_files(root/'labels',root/'displaced',ids,root/'displaced_eval')
        for metric in ('2d','bev','3d'):
            assert all(abs(v-100)<1e-5 for v in perfect['car_AP_R40_percent'][metric].values()),perfect
            expected = 100 if metric=='2d' else 0
            assert all(abs(v-expected)<1e-5 for v in displaced['car_AP_R40_percent'][metric].values()),displaced
        report = dict(state='PASS',perfect=perfect,displaced_3d=displaced,
                      note='Synthetic evaluator fixtures, not detector accuracy.')
    output = Path('outputs/kitti_precheck/evaluator_check.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
