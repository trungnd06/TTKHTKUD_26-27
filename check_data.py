"""Validate the included processed EEG against its published manifest."""
import bootstrap
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'data/processed_manifest.json').read_text(encoding='utf-8'))
    path = ROOT / manifest['file']
    if not path.is_file():
        raise SystemExit('Missing data/epochs.npz. Clone the full repository, or download and preprocess raw data.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise SystemExit('Processed-data SHA-256 differs from the committed pilot. Restore the file, or use the full workflow for new preprocessing.')
    with np.load(path, allow_pickle=False) as d:
        assert list(d['X'].shape) == manifest['shape']
        assert np.isfinite(d['X']).all()
        assert len(d['y']) == len(d['subjects']) == len(d['X'])
        assert np.unique(d['subjects']).tolist() == manifest['subjects']
        assert d['channels'].tolist() == manifest['channels']
        assert int((d['y'] == 0).sum()) == manifest['n_standard']
        assert int((d['y'] == 1).sum()) == manifest['n_target']
    print(f'PASS: processed EEG checksum, shape {manifest["shape"]}, labels and subjects match the committed pilot.')

if __name__ == '__main__':
    main()
