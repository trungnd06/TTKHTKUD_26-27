"""Continuous, label-independent preprocessing and single-trial epoching."""
import bootstrap
import argparse
import json
from pathlib import Path
import mne
import numpy as np

CHANNELS = ['Fz', 'F3', 'F4', 'FC1', 'FC2', 'Cz', 'C3', 'C4',
            'CP1', 'CP2', 'CPz', 'Pz', 'P3', 'P4', 'O1', 'O2']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    p.add_argument('--output', type=Path, default=Path('data/epochs.npz'))
    args = p.parse_args()
    paths = sorted(args.root.glob('sub-*/eeg/*task-oddball_eeg.vhdr'))
    if not paths:
        raise SystemExit('No oddball data. Run python download_data.py first.')
    arrays, labels, subjects, audits = [], [], [], []
    mne.set_log_level('ERROR')
    for path in paths:
        sid = path.name.split('_')[0]
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose=False)
        # Restrict sensors BEFORE loading: <200 MB instead of >1 GB per subject.
        raw.pick(CHANNELS).reorder_channels(CHANNELS).load_data()
        raw.set_montage('standard_1005', on_missing='raise')
        raw.set_eeg_reference('average', projection=False)
        raw.filter(0.5, 30., method='iir', iir_params={'order': 4, 'ftype': 'butter'},
                   skip_by_annotation=('edge', 'bad'))
        events, event_map = mne.events_from_annotations(raw, verbose=False)
        # BrainVision label strings include prefixes and variable whitespace.
        standard = [v for k, v in event_map.items() if k.split('/')[-1].replace(' ', '') == 'S5']
        target = [v for k, v in event_map.items() if k.split('/')[-1].replace(' ', '') == 'S6']
        if len(standard) != 1 or len(target) != 1:
            raise ValueError(f'{sid}: unexpected marker map {event_map}')
        event_id = {'standard': standard[0], 'target': target[0]}
        counts_before = {k: int((events[:, 2] == v).sum()) for k, v in event_id.items()}
        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.,
                            baseline=(-0.2, 0), reject={'eeg': 150e-6},
                            flat={'eeg': 0.1e-6}, preload=True, verbose=False)
        epochs.resample(128, verbose=False)
        X = (epochs.get_data(copy=True) * 1e6).astype('float32')
        y = (epochs.events[:, 2] == target[0]).astype('int64')
        arrays.append(X)
        labels.append(y)
        subjects.extend([sid] * len(y))
        audit = {'subject': sid, 'before': counts_before,
                 'after': {'standard': int((y == 0).sum()), 'target': int(y.sum())},
                 'dropped': int(sum(counts_before.values()) - len(y)),
                 'source_duration_s': float(raw.times[-1]),
                 'event_map': event_map}
        if min(audit['after'].values()) < 10:
            raise ValueError(f'Insufficient clean epochs: {audit}')
        audits.append(audit)
        print(json.dumps(audit), flush=True)
        times = epochs.times
        del raw, epochs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=np.concatenate(arrays), y=np.concatenate(labels),
                        subjects=np.array(subjects), times=times, channels=CHANNELS,
                        sfreq=128., provenance='real OpenNeuro ds004621 oddball')
    (args.output.parent / 'preprocessing_audit.json').write_text(json.dumps(audits, indent=2))
    print(f'Saved {args.output}: {sum(map(len, labels))} real EEG trials', flush=True)

if __name__ == '__main__':
    main()
