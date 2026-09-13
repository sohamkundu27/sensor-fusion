#!/usr/bin/env python3
"""Extract checksum-pinned NVIDIA CUDA development packages without sudo."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default=str(Path.home() / 'venvs/transfusion'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    packages = json.loads((root / 'research/cuda-packages.json').read_text())
    cache = Path.home() / '.cache/transfusion-cuda-debs'
    cache.mkdir(parents=True, exist_ok=True)
    destination = Path(args.prefix).expanduser() / 'cuda-debs'
    destination.mkdir(parents=True, exist_ok=True)
    base = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/'
    for package in packages:
        name = Path(package['Filename']).name
        path = cache / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != package['SHA256']:
            subprocess.run(['curl', '-fL', '--retry', '5', base + name, '-o', str(path)], check=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != package['SHA256']:
            raise RuntimeError('Checksum mismatch: ' + name)
        subprocess.run(['dpkg-deb', '-x', str(path), str(destination)], check=True)
    print('Installed compiler/headers under', destination / 'usr/local/cuda-11.0')
    print('CUDA math-library runtimes are supplied by the pinned PyTorch wheel. No driver was installed.')


if __name__ == '__main__':
    main()
