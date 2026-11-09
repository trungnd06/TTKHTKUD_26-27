"""Check result integrity against held-out predictions and the stored data."""
import bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import metrics, CompactEEGCNN, integrated_gradients
import torch

def main():
    out = Path('results')
    data = np.load('data/epochs.npz')
    predictions = pd.read_csv(out / 'predictions.csv')
    scores = pd.read_csv(out / 'fold_metrics.csv')
    splits = json.loads((out / 'splits.json').read_text())
    checks = []
    assert np.isfinite(data['X']).all() and len(np.unique(data['y'])) == 2
    checks.append('Finite real epochs; two nonempty classes')
    for s in splits:
        train, val, test = map(set, [s['train'], s['validation'], s['test']])
        assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
        assert train | val | test == set(data['subjects'])
    checks.append('Train, validation and test subjects are disjoint in every fold')
    for name, p in predictions.groupby('model'):
        assert len(p) == len(data['y']) and p.trial.nunique() == len(p)
        assert np.array_equal(p.y.to_numpy(), data['y'][p.trial])
        assert np.array_equal(p.subject.to_numpy(), data['subjects'][p.trial])
        assert p.probability.between(0, 1).all()
        for fold, q in p.groupby('fold'):
            assert set(q.subject) == set(splits[fold - 1]['test'])
            expected = scores[(scores.model == name) & (scores.fold == fold)].iloc[0]
            for key, value in metrics(q.y, q.probability).items():
                assert np.isclose(value, expected[key]), (name, fold, key)
    checks.append('Every trial predicted once per model; labels and metrics independently recomputed')
    majority = scores[scores.model == 'Majority']
    assert np.allclose(majority.roc_auc, .5) and np.allclose(majority.balanced_accuracy, .5)
    checks.append('Majority baseline yields AUC=0.5 and balanced accuracy=0.5')
    # Explainability numerical check on an actual trained network and held-out trials.
    torch.set_num_threads(2)
    checkpoint = torch.load(out / 'models/cnn_fold1.pt', weights_only=False, map_location='cpu')
    model = CompactEEGCNN(len(data['channels']))
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    select = np.flatnonzero(np.isin(data['subjects'], checkpoint['split']['test']))[:4]
    x = np.ascontiguousarray((data['X'][select] - checkpoint['mean']) / checkpoint['std'])
    ig = integrated_gradients(model, x, steps=256)
    with torch.no_grad():
        delta = (model(torch.from_numpy(x)) - model(torch.zeros_like(torch.from_numpy(x)))).numpy()
    relative_error = float(np.max(np.abs(ig.sum((1, 2)) - delta) / (np.abs(delta) + .1)))
    assert relative_error < .08, relative_error
    checks.append(f'Integrated-gradients completeness: maximum stabilized relative error {relative_error:.5f}')
    (out / 'verification.json').write_text(json.dumps({'passed': True, 'checks': checks}, indent=2))
    print('\n'.join('PASS: ' + c for c in checks))

if __name__ == '__main__':
    main()
