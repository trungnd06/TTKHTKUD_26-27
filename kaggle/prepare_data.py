"""Attempt all 42 oddball subjects, using bounded temporary raw storage."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import mne
import numpy as np
import requests

BASE = 'https://s3.amazonaws.com/openneuro.org/ds004621/'
CHANNELS = ['Fz', 'F3', 'F4', 'FC1', 'FC2', 'Cz', 'C3', 'C4',
            'CP1', 'CP2', 'CPz', 'Pz', 'P3', 'P4', 'O1', 'O2']
SETTINGS = {'channels': CHANNELS, 'filter_hz': [.5, 30], 'iir_order': 4,
            'epoch_s': [-.2, 1.], 'baseline_s': [-.2, 0], 'sampling_hz': 128,
            'reject_uv': 150, 'flat_uv': .1, 'min_trials_per_class': 10,
            'reference': 'average of selected 16 channels', 'version': 1}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

def download(relative, directory):
    path = Path(directory) / Path(relative).name
    for attempt in range(3):
        try:
            with requests.get(BASE + relative, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                size = int(r.headers.get('Content-Length', 0))
                with path.open('wb') as f:
                    for block in r.iter_content(4*1024*1024):
                        f.write(block)
                if size and path.stat().st_size != size:
                    raise IOError('Incomplete response: ' + relative)
                return path, {'file': relative, 'url': BASE + relative, 'bytes': path.stat().st_size,
                              'etag': r.headers.get('ETag'), 'sha256': digest(path)}
        except (requests.RequestException, IOError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)

def process_subject(header, sid):
    raw = mne.io.read_raw_brainvision(header, preload=False, verbose=False)
    raw.pick(CHANNELS).reorder_channels(CHANNELS).load_data()
    raw.set_montage('standard_1005', on_missing='raise')
    # Split filtering and reject epochs across interior acquisition discontinuities.
    boundaries = [float(a['onset']) for a in raw.annotations
                  if ('new segment' in a['description'].lower() or 'boundary' in a['description'].lower())
                  and .01 < a['onset'] < raw.times[-1] - .01]
    for onset in boundaries:
        raw.annotations.append(onset, .002, 'BAD_ACQ_SKIP')
    raw.set_eeg_reference('average', projection=False)
    raw.filter(.5, 30., method='iir', iir_params={'order': 4, 'ftype': 'butter'},
               skip_by_annotation=('edge', 'bad'), verbose=False)
    events, mapping = mne.events_from_annotations(raw, verbose=False)
    codes = {}
    for label, marker in [('standard', 'S5'), ('target', 'S6')]:
        matches = [v for k, v in mapping.items() if k.split('/')[-1].replace(' ', '') == marker]
        if len(matches) != 1:
            raise ValueError(f'{sid}: cannot identify {marker} in {mapping}')
        codes[label] = matches[0]
    before = {k: int((events[:, 2] == v).sum()) for k, v in codes.items()}
    ep = mne.Epochs(raw, events, codes, tmin=-.2, tmax=1., baseline=(-.2, 0),
                    reject={'eeg': 150e-6}, flat={'eeg': .1e-6},
                    reject_by_annotation=True, preload=True, verbose=False)
    if len(ep):
        ep.resample(128, verbose=False)
        X = (ep.get_data(copy=True) * 1e6).astype('float32')
    else:
        X = np.empty((0, len(CHANNELS), 154), dtype='float32')
    y = (ep.events[:, 2] == codes['target']).astype('int64')
    audit = {'subject': sid, 'before': before,
             'after': {'standard': int((y == 0).sum()), 'target': int((y == 1).sum())},
             'dropped': int(sum(before.values()) - len(y)), 'interior_boundaries_s': boundaries,
             'source_duration_s': float(raw.times[-1]), 'event_map': mapping,
             'behavioral_compliance_flag_from_paper': sid in ['sub-02', 'sub-14', 'sub-17']}
    return X, y, ep.times.copy(), audit

def combine(output, requested):
    audits, arrays, labels, groups = [], [], [], []
    times = None
    for sid in requested:
        audit = json.loads((output / f'{sid}.json').read_text(encoding='utf-8'))
        audits.append(audit)
        if audit['status'] == 'excluded_quality':
            continue
        path = output / f'{sid}.npz'
        if digest(path) != audit['processed_sha256']:
            raise ValueError(f'{sid}: processed checksum mismatch')
        with np.load(path, allow_pickle=False) as d:
            if times is not None:
                np.testing.assert_allclose(times, d['times'])
            times = d['times'].copy()
            arrays.append(d['X'].copy()); labels.append(d['y'].copy())
            groups.extend([sid]*len(d['y']))
    if not arrays:
        raise ValueError('No subjects passed quality criteria.')
    X, y = np.concatenate(arrays), np.concatenate(labels)
    destination = output / 'epochs_full.npz'
    with destination.with_suffix('.npz.partial').open('wb') as f:
        np.savez_compressed(f, X=X, y=y, subjects=np.array(groups), channels=CHANNELS,
                            times=times, sfreq=128., provenance='real OpenNeuro ds004621 oddball')
    destination.with_suffix('.npz.partial').replace(destination)
    summary = {'source': BASE, 'requested_subjects': requested,
               'included_subjects': sorted(set(groups)),
               'excluded_subjects': [a['subject'] for a in audits if a['status'] != 'included'],
               'all_42_attempted': requested == [f'sub-{i:02d}' for i in range(1, 43)],
               'settings': SETTINGS, 'n_trials': len(y), 'n_standard': int((y == 0).sum()),
               'n_target': int((y == 1).sum()), 'shape': list(X.shape),
               'processed_sha256': digest(destination), 'audits': audits,
               'mne': mne.__version__, 'numpy': np.__version__}
    write_json(output / 'preparation_summary.json', summary)
    print(f'COMPLETE: requested {len(requested)}, retained {len(set(groups))} subjects; {len(y)} trials', flush=True)
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('/kaggle/working/nencki_full/processed'))
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 43)))
    p.add_argument('--raw-root', type=Path, help='Optional local BIDS root; input files are never removed')
    args = p.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    cache = output.parent / '.raw-cache'; cache.mkdir(exist_ok=True)
    requested = [f'sub-{i:02d}' for i in sorted(set(args.subjects))]
    if any(not 1 <= i <= 42 for i in args.subjects):
        p.error('Subjects must be between 1 and 42.')
    mne.set_log_level('ERROR')
    # Save source descriptions alongside the processed dataset (once).
    for name in ['README', 'dataset_description.json', 'task-oddball_eeg.json', 'task-oddball_events.json']:
        if not (output / name).exists():
            if args.raw_root and (args.raw_root / name).exists():
                (output / name).write_bytes((args.raw_root / name).read_bytes())
            elif not args.raw_root:
                download(name, output)
    for index, sid in enumerate(requested, 1):
        audit_path = output / f'{sid}.json'
        if audit_path.exists():
            a = json.loads(audit_path.read_text(encoding='utf-8'))
            if a['settings'] != SETTINGS:
                raise ValueError('Settings changed; use a new output directory.')
            if a['status'] == 'excluded_quality' or (output / f'{sid}.npz').exists():
                print(f'RESUME {index}/{len(requested)} {sid}', flush=True)
                continue
        print(f'PROCESS {index}/{len(requested)} {sid}', flush=True)
        try:
            if args.raw_root:
                header = args.raw_root / sid / 'eeg' / f'{sid}_task-oddball_eeg.vhdr'
                X, y, times, audit = process_subject(header, sid)
                provenance = []
                for suffix in ['eeg.eeg', 'eeg.vhdr', 'eeg.vmrk', 'events.tsv', 'channels.tsv']:
                    path = header.parent / f'{sid}_task-oddball_{suffix}'
                    if not path.exists():
                        raise FileNotFoundError(path)
                    provenance.append({'file': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size})
            else:
                # A directory created and owned by this script; never delete an input dataset.
                with tempfile.TemporaryDirectory(prefix=sid+'-', dir=cache) as temporary:
                    assert Path(temporary).resolve().is_relative_to(cache.resolve())
                    provenance = []
                    for suffix in ['eeg.vhdr', 'eeg.vmrk', 'events.tsv', 'channels.tsv', 'eeg.eeg']:
                        _, entry = download(f'{sid}/eeg/{sid}_task-oddball_{suffix}', temporary)
                        provenance.append(entry)
                    X, y, times, audit = process_subject(Path(temporary) / f'{sid}_task-oddball_eeg.vhdr', sid)
            audit.update(settings=SETTINGS, files=provenance,
                         environment={'mne': mne.__version__, 'numpy': np.__version__})
            if min(audit['after'].values()) < SETTINGS['min_trials_per_class']:
                audit.update(status='excluded_quality', reason='fewer than 10 retained trials in a class')
            else:
                path = output / f'{sid}.npz'
                with path.with_suffix('.npz.partial').open('wb') as f:
                    np.savez_compressed(f, X=X, y=y, times=times, channels=CHANNELS)
                path.with_suffix('.npz.partial').replace(path)
                audit.update(status='included', processed_sha256=digest(path))
            write_json(audit_path, audit)
            print(f'{sid}: {audit["status"]} {audit["after"]}', flush=True)
        except Exception as error:
            write_json(output / 'last_error.json', {'subject': sid, 'error': str(error)})
            raise  # Never silently treat network/parsing failures as quality exclusions.
    combine(output, requested)

if __name__ == '__main__':
    main()
