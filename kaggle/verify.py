"""Reconstruct full-run metrics and check held-out subject coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train import metrics, CompactEEGCNN, integrated_gradients
from prepare_data import digest, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    d = np.load(args.data, allow_pickle=False)
    out = args.output
    metadata = json.loads((out / 'run_metadata.json').read_text(encoding='utf-8'))
    assert digest(args.data) == metadata['data_sha256']
    preds = pd.read_csv(out / 'predictions.csv')
    scores = pd.read_csv(out / 'fold_metrics.csv')
    splits = json.loads((out / 'splits.json').read_text(encoding='utf-8'))
    checks = []
    assert set(preds.model) == {'Majority', 'Shrinkage LDA', 'RBF SVM', 'Random Forest', 'Compact EEG CNN'}
    assert len(splits) == metadata['folds']
    for s in splits:
        tr, va, te = (set(s[k]) for k in ['train', 'validation', 'test'])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        assert tr | va | te == set(d['subjects'])
    checks.append('Disjoint train/validation/test subjects in all folds')
    for name, pp in preds.groupby('model'):
        assert len(pp) == len(d['y']) == pp.trial.nunique()
        assert set(pp.fold) == set(range(1, metadata['folds'] + 1))
        assert np.array_equal(pp.y.to_numpy(), d['y'][pp.trial])
        assert np.array_equal(pp.subject.to_numpy(), d['subjects'][pp.trial])
        assert pp.probability.between(0, 1).all()
        for fold, q in pp.groupby('fold'):
            assert set(q.subject) == set(splits[fold-1]['test'])
            expected = scores[(scores.model == name) & (scores.fold == fold)]
            assert len(expected) == 1
            for metric, value in metrics(q.y, q.probability).items():
                assert np.isclose(value, expected.iloc[0][metric]), (name, fold, metric)
    checks.append('Each retained trial tested exactly once for every model; metrics reconstructed')
    base = scores[scores.model == 'Majority']
    assert np.allclose(base.roc_auc, .5) and np.allclose(base.balanced_accuracy, .5)
    checks.append('Majority AUC and balanced accuracy both equal 0.5')
    torch.set_num_threads(2)
    c = torch.load(out / 'models/cnn_fold1.pt', map_location='cpu', weights_only=False)
    model = CompactEEGCNN(len(d['channels'])); model.load_state_dict(c['state_dict']); model.eval()
    ix = np.flatnonzero(np.isin(d['subjects'], c['split']['test']))[:4]
    x = np.ascontiguousarray((d['X'][ix] - c['mean']) / c['std'])
    ig = integrated_gradients(model, x, steps=256)
    with torch.no_grad():
        delta = (model(torch.from_numpy(x)) - model(torch.zeros_like(torch.from_numpy(x)))).numpy()
    error = float(np.max(np.abs(ig.sum((1, 2)) - delta) / (np.abs(delta) + .1)))
    assert error < .08, error
    checks.append(f'Integrated-gradients completeness, maximum stabilized relative error {error:.6f}')
    write_json(out / 'verification.json', {'passed': True, 'checks': checks})
    avg = scores.groupby('model')[['roc_auc', 'balanced_accuracy', 'average_precision', 'f1']].agg(['mean', 'std'])
    best = scores[scores.model != 'Majority'].groupby('model').roc_auc.mean().idxmax()
    report = ['# Measured EEG classification results',
              f'Full cohort attempted: {metadata["full_cohort_attempted"]}. Retained subjects: {len(metadata["subjects"])}.',
              f'Quality exclusions: {metadata["quality_excluded_subjects"]}. Trials: {metadata["n_trials"]}.',
              f'Highest observed mean ROC-AUC among tested models: {best}.',
              '```text\n' + avg.to_string() + '\n```',
              'Mean +/- SD across subject-disjoint folds. SD is not a confidence interval. '
              'Model ranking is descriptive. One base seed; residual artifacts, visual and motor-response confounds remain.',
              'The CNN uses the full -0.2 to 1.0 s epoch; classical features use 0 to 1.0 s. '
              'Attribution and occlusion show model sensitivity, not causal neural localization.',
              'Behavioral-compliance flags (sub-02, sub-14, sub-17) are retained if signal quality permits. '
              'They are not silently excluded; see preparation_summary.json.',
              'See fold_metrics.csv, predictions.csv, run_metadata.json and splits.json for exact details. '
              'The original report in the parent project concerns the eight-subject pilot and is not updated by this workflow.']
    (out / 'RESULTS_EN.md').write_text('\n\n'.join(report), encoding='utf-8')
    print('\n'.join('PASS: '+c for c in checks))

if __name__ == '__main__':
    main()
