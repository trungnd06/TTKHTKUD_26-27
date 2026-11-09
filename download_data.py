"""Download only oddball recordings from the public OpenNeuro S3 mirror."""
import bootstrap
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

BASE = 'https://s3.amazonaws.com/openneuro.org/ds004621/'
DEFAULT_SUBJECTS = ['01', '03', '04', '05', '06', '07', '08', '09']

def fetch(relative, root):
    dest = root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + relative
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                size = int(response.headers.get('Content-Length', 0))
                if not dest.exists() or (size and dest.stat().st_size != size):
                    partial = dest.with_suffix(dest.suffix + '.partial')
                    with partial.open('wb') as f:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            f.write(chunk)
                    if size and partial.stat().st_size != size:
                        raise IOError('Incomplete download: ' + relative)
                    partial.replace(dest)
            digest = hashlib.sha256()
            with dest.open('rb') as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            print(f'OK {relative} ({dest.stat().st_size:,} bytes)', flush=True)
            return {'file': relative, 'url': url, 'bytes': dest.stat().st_size,
                    'sha256': digest.hexdigest(), 'etag': response.headers.get('ETag')}
        except (requests.RequestException, IOError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    args = p.parse_args()
    files = ['README', 'dataset_description.json', 'task-oddball_eeg.json',
             'task-oddball_events.json']
    for subject in args.subjects:
        sid = f'sub-{int(subject):02d}'
        stem = f'{sid}/eeg/{sid}_task-oddball_'
        files += [stem + suffix for suffix in
                  ['eeg.vhdr', 'eeg.vmrk', 'eeg.eeg', 'events.tsv', 'channels.tsv']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(lambda f: fetch(f, args.root), files))
    args.root.parent.mkdir(parents=True, exist_ok=True)
    (args.root.parent / 'download_manifest.json').write_text(
        json.dumps({'dataset': 'ds004621', 'source': BASE, 'files': entries}, indent=2),
        encoding='utf-8')

if __name__ == '__main__':
    main()
