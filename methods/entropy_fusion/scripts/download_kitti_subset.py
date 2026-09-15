"""Download a bounded KITTI camera/LiDAR subset from public archives using HTTP ranges.

ZIP CRCs are checked on every extracted file. No unlabeled test data is downloaded.
The train/validation lists come from a pinned OpenPCDet revision.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import time
import urllib.request
import zipfile
import zlib

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'
ARCHIVES = {'image_2': 'png', 'velodyne': 'bin', 'calib': 'txt', 'label_2': 'txt'}


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.transferred = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers['ETag']
        self.cache_start, self.cache = 0, b''

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if whence not in (0,1,2): raise ValueError('Invalid seek mode')
        position = offset+(0,self.position,self.size)[whence]
        if position < 0: raise ValueError('Invalid seek')
        self.position = position
        return position

    def read(self, count=-1):
        count = max(0, self.size-self.position) if count < 0 else min(count, max(0,self.size-self.position))
        if count == 0: return b''
        if count > 64*1024**2: raise ValueError('Refusing unbounded archive read')
        start, end = self.position, self.position+count
        if not (self.cache_start <= start and end <= self.cache_start+len(self.cache)):
            stop = min(self.size, max(end, start+1024**2))
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{stop-1}', 'If-Match': self.etag})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop-1}/{self.size}':
                    raise IOError('Server did not honor exact byte range; refusing full archive download')
                data = response.read(stop-start+1)
            if len(data) != stop-start: raise IOError('Truncated range response')
            self.transferred += len(data)
            self.cache_start, self.cache = start, data
        self.position = end
        return self.cache[start-self.cache_start:end-self.cache_start]


def fetch_archive(root, folder, extension, ids):
    url = BASE+f'data_object_{folder}.zip'
    reader = RemoteZipReader(url)
    records = []
    started = time.monotonic()
    with zipfile.ZipFile(reader) as archive:
        for index, sample in enumerate(ids):
            name = f'training/{folder}/{sample}.{extension}'
            info = archive.getinfo(name)
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                data = path.read_bytes()
                if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                    raise IOError(f'Existing file fails CRC: {path}')
            else:
                data = archive.read(info)  # ZipFile checks decompression/CRC before returning.
                temporary = path.with_suffix(path.suffix+'.tmp')
                temporary.write_bytes(data)
                temporary.replace(path)
            records.append(dict(path=name, bytes=len(data), crc32=f'{info.CRC:08x}', sha256=hashlib.sha256(data).hexdigest()))
            if (index+1) % 10 == 0 or index+1==len(ids):
                progress = dict(folder=folder,verified=index+1,total=len(ids),
                                transferred_bytes=reader.transferred,elapsed_seconds=time.monotonic()-started,
                                state='completed' if index+1==len(ids) else 'downloading')
                temporary = root/f'{folder}_progress.tmp'
                temporary.write_text(json.dumps(progress,indent=2)+'\n')
                temporary.replace(root/f'{folder}_progress.json')
                print(f'{folder}: verified {index+1}/{len(ids)}', flush=True)
    return dict(url=url, etag=reader.etag, archive_bytes=reader.size, transferred_bytes=reader.transferred, files=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home()/'data/kitti')
    parser.add_argument('--train-count', type=int, default=32)
    parser.add_argument('--val-count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.train_count,args.val_count) < 1: parser.error('Positive subset sizes required')
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'subset_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if (old['train_count'],old['val_count'],old['seed']) != (args.train_count,args.val_count,args.seed):
            raise ValueError('Choose a different root for a different subset')
        commit = old['split_commit']
    else:
        with urllib.request.urlopen('https://api.github.com/repos/open-mmlab/OpenPCDet/commits/master', timeout=60) as r:
            commit = json.load(r)['sha']
    subsets, source_hashes = {}, {}
    for split, count in [('train',args.train_count),('val',args.val_count)]:
        url = f'https://raw.githubusercontent.com/open-mmlab/OpenPCDet/{commit}/data/kitti/ImageSets/{split}.txt'
        with urllib.request.urlopen(url, timeout=60) as r: raw = r.read()
        all_ids = raw.decode().splitlines()
        if len(set(all_ids)) != len(all_ids) or not all(re.fullmatch(r'\d{6}', x) for x in all_ids):
            raise ValueError('Invalid split file')
        subsets[split] = sorted(random.Random(args.seed).sample(all_ids,count))
        source_hashes[split] = hashlib.sha256(raw).hexdigest()
        (root/'ImageSets').mkdir(exist_ok=True)
        (root/'ImageSets'/f'{split}_full.txt').write_bytes(raw)
        (root/'ImageSets'/f'{split}.txt').write_text('\n'.join(subsets[split])+'\n')
    if set(subsets['train']) & set(subsets['val']): raise ValueError('Training/validation overlap')
    report = dict(state='downloading', train_count=args.train_count, val_count=args.val_count, seed=args.seed,
                  split_commit=commit, split_sha256=source_hashes, subsets=subsets, archives=[])
    manifest_path.write_text(json.dumps(report,indent=2)+'\n')
    ids = sorted(subsets['train']+subsets['val'])
    try:
        # Conservative budget for extracted frames and range buffers; no full ZIP
        # copies are retained. The actual bytes are reported after verification.
        if shutil.disk_usage(root).free < len(ids)*8*1024**2+1024**3:
            raise OSError('Insufficient free disk for conservative 8 MiB/frame budget')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch_archive,root,folder,ext,ids) for folder,ext in ARCHIVES.items()]
            report['archives'] = [future.result() for future in futures]
        report['state'] = 'verified'
        report['extracted_bytes'] = sum(f['bytes'] for a in report['archives'] for f in a['files'])
    except Exception as error:
        report.update(state='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['state','train_count','val_count','extracted_bytes']},indent=2))


if __name__ == '__main__': main()
