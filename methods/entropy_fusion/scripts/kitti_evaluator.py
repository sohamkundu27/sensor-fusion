"""Local wrapper around the checksum-pinned KITTI C++ 2D/BEV/3D evaluator.

Changes are limited to dataset length, local entry point, stdout logging and
disabled plotting. Matching, difficulty filters and precision curves are upstream.
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile
import numpy as np

URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip'
SHA256 = 'ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1'


def build_evaluator(count, tools=None):
    if count < 1: raise ValueError('Nonempty evaluation set required')
    tools = Path(tools or Path.home()/'data/kitti_tools')
    tools.mkdir(parents=True,exist_ok=True)
    archive = tools/'devkit_object.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=60) as response: raw = response.read(5*1024**2)
        if hashlib.sha256(raw).hexdigest()!=SHA256: raise ValueError('Unexpected official devkit checksum')
        archive.write_bytes(raw)
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA256: raise ValueError('Devkit checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        source = z.read('cpp/evaluate_object.cpp').decode()
        (tools/'upstream_evaluate_object.cpp').write_text(source)
        (tools/'upstream_readme.txt').write_bytes(z.read('readme.txt'))
    old = 'const int32_t N_TESTIMAGES = 7518;'
    assert source.count(old)==1
    source = source.replace(old,f'const int32_t N_TESTIMAGES = {count};')
    source = source[:source.index('int32_t main (')]+'''int main() {
      Mail log;
      return eval("local", &log) ? 0 : 1;
    }
'''
    source = source.replace('#include "mail.h"','''#include <cstdarg>
class Mail { public: void msg(const char *format, ...) {
  va_list args; va_start(args, format); vprintf(format, args); va_end(args); printf("\\n");
}};
''')
    source = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('saveAndPlotPlots('))+'\n'
    digest = hashlib.sha256(source.encode()).hexdigest()
    source_path = tools/f'local_evaluate_{count}_{digest[:12]}.cpp'
    binary = source_path.with_suffix('')
    if not binary.exists():
        source_path.write_text(source)
        command = ['g++','-O2','-std=c++14',str(source_path),'-o',str(binary)]
        if not Path('/usr/include/boost/geometry.hpp').exists():
            include = tools/'deps/usr/include'
            if not (include/'boost/geometry.hpp').exists(): raise FileNotFoundError('Install/extract Ubuntu libboost-dev headers; see KITTI_EXPERIMENT.md')
            command += ['-I',str(include)]
        with (tools/'build.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    return binary,dict(upstream_url=URL,upstream_sha256=SHA256,patched_source_sha256=digest,frames=count)


def evaluate_files(label_root, detection_root, ids, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(f'Use a fresh evaluation directory: {output}')
    binary,provenance = build_evaluator(len(ids))
    gt = output/'data/object/label_2'
    result = output/'results/local'
    gt.mkdir(parents=True)
    (result/'data').mkdir(parents=True)
    detections = 0
    for index,sample in enumerate(ids):
        shutil.copyfile(Path(label_root)/f'{sample}.txt',gt/f'{index:06d}.txt')
        source = Path(detection_root)/f'{sample}.txt'
        detections += len(source.read_text().splitlines())
        shutil.copyfile(source,result/'data'/f'{index:06d}.txt')
    (output/'frame_mapping.json').write_text(json.dumps(ids,indent=2)+'\n')
    with (output/'evaluator.log').open('w') as log:
        subprocess.run([str(binary)],cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics = {}
    for metric,suffix in [('2d',''),('bev','_ground'),('3d','_3d')]:
        path = result/f'stats_car_detection{suffix}.txt'
        if not path.exists() and detections==0:
            curve = np.zeros((3,41))
        else:
            curve = np.loadtxt(path)
        if curve.shape != (3,41) or not np.isfinite(curve).all(): raise ValueError('Invalid evaluator precision curves')
        metrics[metric] = {difficulty:float(curve[i,1:].mean()*100) for i,difficulty in enumerate(('easy','moderate','hard'))}
    report = dict(car_AP_R40_percent=metrics,provenance=provenance,metric='KITTI car IoU=0.7; R40 average of precision positions 1..40',detections=detections)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
