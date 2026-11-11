"""Subject-disjoint ML/CNN comparison with out-of-fold explanations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
from prepare_data import digest, write_json
import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CompactEEGCNN(nn.Module):
    """EEGNet-inspired model; no claim of an exact EEGNet reproduction."""
    def __init__(self, channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 33), padding=(0, 16), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(.35),
            nn.Conv2d(16, 16, (1, 9), padding=(0, 4), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)), nn.Dropout(.35))
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.features(x[:, None]).flatten(1)).squeeze(1)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def metrics(y, p):
    pred = p >= .5
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred))

def erp_features(X, times):
    # Twenty 50-ms means per channel over [0, 1] seconds; no test-fitted transform.
    return np.concatenate([X[:, :, (times >= a) & (times < a + .05)].mean(-1)
                           for a in np.arange(0., 1., .05)], axis=1)

def predict_cnn(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(x).to(next(model.parameters()).device)).sigmoid().cpu()
                          for x in np.array_split(X, max(1, int(np.ceil(len(X) / 128))))]).numpy()

def train_cnn(X, y, V, vy, epochs, seed, device):
    seed_all(seed)
    model = CompactEEGCNN(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / y.sum()), device=device))
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype('float32'))),
                        batch_size=64, shuffle=True)
    best_auc, best_epoch, best_state = -1, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            losses.append(float(loss.detach()))
        auc = roc_auc_score(vy, predict_cnn(model, V))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_auc': auc})
        print(f'  CNN epoch {epoch}: validation AUC {auc:.4f}', flush=True)
        if auc > best_auc + .0001:
            best_auc, best_epoch, best_state = auc, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    return model, best_auc, best_epoch, history

def integrated_gradients(model, X, steps=24):
    # Zero baseline is the TRAINING channel mean after standardization.
    model.eval()
    inp = torch.from_numpy(X).to(next(model.parameters()).device)
    grads = torch.zeros_like(inp)
    # Midpoint quadrature approximates integral of target logit gradient.
    for alpha in (np.arange(steps) + .5) / steps:
        xi = (inp * float(alpha)).requires_grad_(True)
        grads += torch.autograd.grad(model(xi).sum(), xi)[0].detach()
    return (inp * grads / steps).cpu().numpy()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/kaggle/working/nencki_full/processed/epochs_full.npz'))
    p.add_argument('--output', type=Path, default=Path('/kaggle/working/nencki_full/results'))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--validation-fraction', type=float, default=.2)
    p.add_argument('--allow-partial', action='store_true', help='Explicitly allow a smoke test instead of the full cohort')
    p.add_argument('--overwrite', action='store_true', help='Recompute cached folds in this output directory')
    args = p.parse_args()
    if not 0 < args.validation_fraction < 1 or args.epochs < 1 or args.folds < 2:
        p.error('Invalid training configuration.')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Enable Kaggle GPU, or select cpu.')
    print(f'CNN device: {device}; classical ML uses CPU', flush=True)
    torch.set_num_threads(2)
    seed_all(args.seed)
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / 'models').mkdir(exist_ok=True)
    data_sha = digest(args.data)
    prep_path = args.data.parent / 'preparation_summary.json'
    prep = json.loads(prep_path.read_text(encoding='utf-8')) if prep_path.exists() else None
    if not args.allow_partial:
        if not prep or not prep['all_42_attempted'] or len(prep['requested_subjects']) != 42:
            raise ValueError('Full-cohort preparation metadata required. Use --allow-partial only for an explicit smoke test.')
        if prep['processed_sha256'] != data_sha:
            raise ValueError('Data checksum differs from preparation metadata.')
    data = np.load(args.data, allow_pickle=False)
    X, y, groups, times = [data[k] for k in ['X', 'y', 'subjects', 'times']]
    channels = data['channels'].tolist()
    if len(np.unique(groups)) < args.folds + 2 or len(np.unique(y)) != 2:
        raise ValueError('Need >= folds+2 subjects and both labels.')
    if not np.isfinite(X).all() or any(not set(y[groups == g]) == {0, 1} for g in np.unique(groups)):
        raise ValueError('Non-finite data or a subject is missing a class.')
    if not args.allow_partial and set(np.unique(groups)) != set(prep['included_subjects']):
        raise ValueError('Subject list differs from preparation metadata.')
    run_config = {'data_sha256': data_sha, 'epochs': args.epochs, 'folds': args.folds,
                  'seed': args.seed, 'validation_fraction': args.validation_fraction,
                  'device': str(device), 'torch': torch.__version__, 'sklearn': sklearn.__version__,
                  'numpy': np.__version__, 'script_sha256': digest(__file__)}
    fingerprint = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    config_path = args.output / 'run_config.json'
    if config_path.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous != run_config:
            raise ValueError('Run configuration changed. Use a new --output directory or explicit --overwrite.')
    write_json(config_path, run_config)
    (args.output / 'cached_folds').mkdir(exist_ok=True)
    F = erp_features(X, times)
    rows, predictions, splits, histories, explanations, occlusions = [], [], [], [], [], []
    outer = GroupKFold(args.folds)
    for fold, (development, test) in enumerate(outer.split(X, y, groups), 1):
        dev_subjects = np.unique(groups[development])
        if len(dev_subjects) < 4:
            raise ValueError('Each development fold needs at least 4 subjects for 2 train + 2 validation.')
        # Rotate the two validation subjects deterministically, without using labels or scores.
        n_val = min(len(dev_subjects) - 2, max(2, int(np.ceil(len(dev_subjects) * args.validation_fraction))))
        val_subjects = np.roll(dev_subjects, fold - 1)[:n_val]
        val = development[np.isin(groups[development], val_subjects)]
        train = development[~np.isin(groups[development], val_subjects)]
        split = {'fold': fold, 'train': np.unique(groups[train]).tolist(),
                 'validation': np.unique(groups[val]).tolist(), 'test': np.unique(groups[test]).tolist()}
        assert not (set(split['train']) & set(split['test']) or
                    set(split['train']) & set(split['validation']) or
                    set(split['validation']) & set(split['test']))
        splits.append(split)
        print(f'Fold {fold}: {split}', flush=True)
        cache_path = args.output / 'cached_folds' / f'fold{fold}.json'
        if cache_path.exists() and not args.overwrite:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            checkpoint = args.output / 'models' / f'cnn_fold{fold}.pt'
            if cached['fingerprint'] != fingerprint or digest(checkpoint) != cached['checkpoint_sha256']:
                raise ValueError('Cached fold does not match this run.')
            rows.extend(cached['rows']); predictions.extend(cached['predictions'])
            histories.extend(cached['histories']); occlusions.extend(cached['occlusions'])
            explanations.append(np.array(cached['importance']))
            print(f'RESUME completed fold {fold}', flush=True)
            continue
        candidates = {
            'Majority': [DummyClassifier(strategy='prior')],
            'Shrinkage LDA': [make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
                solver='lsqr', shrinkage='auto', priors=[.5, .5]))],
            'RBF SVM': [make_pipeline(StandardScaler(), SVC(C=c, kernel='rbf', class_weight='balanced',
                          probability=True, random_state=args.seed)) for c in [.1, 1., 10.]],
            'Random Forest': [RandomForestClassifier(n_estimators=200, max_depth=depth,
                min_samples_leaf=3, class_weight='balanced', n_jobs=1, random_state=args.seed)
                              for depth in [6, None]]}
        for name, models in candidates.items():
            started = time.perf_counter()
            scored = []
            for model in models:
                model.fit(F[train], y[train])
                scored.append(roc_auc_score(y[val], model.predict_proba(F[val])[:, 1]))
            chosen = int(np.argmax(scored))
            score = models[chosen].predict_proba(F[test])[:, 1]
            row = {'fold': fold, 'model': name, 'validation_auc': scored[chosen],
                   'choice': chosen, 'seconds': time.perf_counter() - started, **metrics(y[test], score)}
            rows.append(row)
            predictions.extend(dict(fold=fold, model=name, trial=int(i), subject=groups[i],
                                    y=int(y[i]), probability=float(s)) for i, s in zip(test, score))
            print(name, json.dumps(row), flush=True)
        # Fit normalization ONLY on training trials, over trial and time axes.
        mu = X[train].mean(axis=(0, 2), keepdims=True)
        sd = X[train].std(axis=(0, 2), keepdims=True).clip(1e-3)
        A, V, T = [np.ascontiguousarray((X[ix] - mu) / sd, dtype='float32') for ix in [train, val, test]]
        started = time.perf_counter()
        model, val_auc, best_epoch, history = train_cnn(A, y[train], V, y[val], args.epochs, args.seed + fold, device)
        score = predict_cnn(model, T)
        row = {'fold': fold, 'model': 'Compact EEG CNN', 'validation_auc': val_auc,
               'choice': best_epoch, 'seconds': time.perf_counter() - started, **metrics(y[test], score)}
        rows.append(row)
        print('CNN', json.dumps(row), flush=True)
        histories.extend(dict(fold=fold, **h) for h in history)
        predictions.extend(dict(fold=fold, model='Compact EEG CNN', trial=int(i), subject=groups[i],
                                y=int(y[i]), probability=float(s)) for i, s in zip(test, score))
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'channels': channels, 'mean': mu, 'std': sd,
                    'split': split, 'epoch': best_epoch}, args.output / 'models' / f'cnn_fold{fold}.pt')
        # Equal number of held-out standard/target trials; deterministic explanation subset.
        rng = np.random.default_rng(args.seed + fold)
        n = min(32, int((y[test] == 0).sum()), int((y[test] == 1).sum()))
        chosen = np.concatenate([rng.choice(np.flatnonzero(y[test] == k), n, replace=False) for k in [0, 1]])
        explanations.append(np.abs(integrated_gradients(model, T[chosen])).mean(0))
        for start in np.arange(0, 1, .1):
            altered = T.copy()
            altered[:, :, (times >= start) & (times < start + .1)] = 0
            occlusions.append({'fold': fold, 'start_s': start,
                               'auc_drop': row['roc_auc'] - roc_auc_score(y[test], predict_cnn(model, altered))})
        payload = {'fingerprint': fingerprint, 'split': split,
                   'checkpoint_sha256': digest(args.output / 'models' / f'cnn_fold{fold}.pt'),
                   'rows': [r for r in rows if r['fold'] == fold],
                   'predictions': [r for r in predictions if r['fold'] == fold],
                   'histories': [r for r in histories if r['fold'] == fold],
                   'occlusions': [r for r in occlusions if r['fold'] == fold],
                   'importance': explanations[-1].tolist()}
        write_json(cache_path, payload)
    df, preds = pd.DataFrame(rows), pd.DataFrame(predictions)
    df.to_csv(args.output / 'fold_metrics.csv', index=False)
    preds.to_csv(args.output / 'predictions.csv', index=False)
    pd.DataFrame(histories).to_csv(args.output / 'training_history.csv', index=False)
    pd.DataFrame(occlusions).to_csv(args.output / 'occlusion.csv', index=False)
    summary = df.groupby('model')[['accuracy', 'balanced_accuracy', 'roc_auc', 'average_precision', 'f1',
                                   'precision', 'recall']].agg(['mean', 'std'])
    summary.to_csv(args.output / 'summary.csv')
    (args.output / 'splits.json').write_text(json.dumps(splits, indent=2))
    metadata = {'source': str(data['provenance']), 'subjects': np.unique(groups).tolist(),
                'n_trials': len(y), 'n_standard': int((y == 0).sum()), 'n_target': int(y.sum()),
                'channels': channels, 'sampling_hz': 128, 'epoch_seconds': [-.2, 1.],
                'seed': args.seed, 'folds': args.folds, 'max_epochs': args.epochs,
                'device': str(device), 'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                'determinism': 'seeded, deterministic algorithms with warn_only=True',
                'validation_fraction': args.validation_fraction, 'data_sha256': data_sha,
                'full_cohort_attempted': bool(prep and prep['all_42_attempted']),
                'quality_excluded_subjects': prep['excluded_subjects'] if prep else [],
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'sklearn': sklearn.__version__, 'torch': torch.__version__}
    (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(args.output / 'attributions.npz', importance=np.mean(explanations, axis=0),
                        fold_importance=np.array(explanations), times=times, channels=channels)
    make_figures(X, y, groups, times, channels, df, preds, explanations, occlusions, args.output)
    print(summary.to_string(), flush=True)

def make_figures(X, y, groups, times, channels, df, preds, explanations, occlusions, out):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, channel in zip(axs, ['Fz', 'Pz']):
        ci = channels.index(channel)
        for label, name, color in [(0, 'Standard', '#64748b'), (1, 'Target', '#dc7633')]:
            # Subject-weighted grand average, shaded BETWEEN-SUBJECT standard error.
            means = np.stack([X[(y == label) & (groups == g), ci].mean(0) for g in np.unique(groups)])
            mean, se = means.mean(0), means.std(0, ddof=1) / np.sqrt(len(means))
            ax.plot(times, mean, label=name, color=color)
            ax.fill_between(times, mean - se, mean + se, alpha=.15, color=color)
        ax.axvline(0, color='black', lw=.7)
        ax.set(xlabel='Time after stimulus (s)', ylabel='Amplitude (microvolt)', title=channel)
        ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / 'erp.png'); plt.close(fig)
    names = df['model'].unique()
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, metric in enumerate(['balanced_accuracy', 'roc_auc']):
        vals = [df[df.model == n][metric] for n in names]
        ax.bar(np.arange(len(names)) + (j - .5) * .34, [v.mean() for v in vals], width=.34,
               yerr=[v.std(ddof=1) for v in vals], capsize=3, label=metric.replace('_', ' ').title())
    ax.set(xticks=np.arange(len(names)), xticklabels=names, ylim=(0, 1.07),
           ylabel='Score (mean +/- fold SD)', title='Unseen-subject classification')
    ax.axhline(.5, color='gray', ls='--', lw=.7); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / 'comparison.png'); plt.close(fig)
    fig, axs = plt.subplots(1, len(names), figsize=(15, 3))
    for ax, name in zip(axs, names):
        p = preds[preds.model == name]
        cm = confusion_matrix(p.y, p.probability >= .5)
        ax.imshow(cm, cmap='Blues')
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha='center', va='center', color='black', fontsize=11)
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=['Std', 'Target'],
               yticklabels=['Std', 'Target'], xlabel='Predicted')
    axs[0].set_ylabel('True'); fig.tight_layout(); fig.savefig(out / 'confusion.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        # Foldwise curves avoid comparing independently calibrated fold scores.
        curves = []
        grid = np.linspace(0, 1, 101)
        for _, p in preds[preds.model == name].groupby('fold'):
            fpr, tpr, _ = roc_curve(p.y, p.probability)
            curves.append(np.interp(grid, fpr, tpr))
        ax.plot(grid, np.mean(curves, axis=0), label=name)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Mean outer-fold ROC')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / 'roc.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    imp = np.mean(explanations, axis=0)
    im = ax.imshow(imp, aspect='auto', extent=[times[0], times[-1], len(channels) - .5, -.5], cmap='magma')
    ax.set(yticks=np.arange(len(channels)), yticklabels=channels, xlabel='Time (s)',
           title='CNN: mean absolute integrated gradients on held-out trials')
    fig.colorbar(im, ax=ax, label='Absolute target-logit attribution')
    fig.tight_layout(); fig.savefig(out / 'attribution.png'); plt.close(fig)
    o = pd.DataFrame(occlusions).groupby('start_s').auc_drop.agg(['mean', 'std'])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(o.index + .05, o['mean'], width=.08, yerr=o['std'], capsize=3)
    ax.axhline(0, color='black', lw=.7)
    ax.set(xlabel='Occluded 100-ms window centre (s)', ylabel='Held-out AUC decrease',
           title='CNN temporal occlusion (mean +/- fold SD)')
    fig.tight_layout(); fig.savefig(out / 'occlusion.png'); plt.close(fig)

if __name__ == '__main__':
    main()
