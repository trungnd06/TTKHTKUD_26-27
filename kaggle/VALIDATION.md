# Verification performed before delivery

Date: 2026-11-11. Local Windows / Python 3.13.5 / PyTorch 2.12.0 CPU environment.

Passed checks:

- All Python scripts compile. All notebook code cells compile as Python; notebooks contain no saved outputs or credentials.
- OpenNeuro metadata download via the new downloader succeeds; S6 is the target marker.
- New preprocessing on the eight already-downloaded real subjects retains 4,543 trials.
  Labels, subjects, channel order and time coordinates match the pilot exactly.
  Maximum absolute signal difference is 0.000003815 microvolts (floating-point roundoff).
- Preprocessing resumes completed subjects and reproduces the combined NPZ checksum.
- Training completes all five classifiers on real EEG: eight-subject smoke test, two outer folds, one CNN epoch per fold.
- Training resumes both completed folds using fingerprint/checkpoint checks, without retraining them.
- Verification reconstructs every reported fold metric, verifies subject-disjoint splits and once-per-trial test coverage.
- Majority baseline ROC-AUC and balanced accuracy equal 0.5.
- Integrated-gradients completeness check passes on held-out real EEG (stabilized relative error approximately 0.000001).
- The default full-cohort guard rejects the eight-subject pilot without explicit --allow-partial.

Not performed here:

- Downloading and processing the other 34 subjects.
- Full-cohort model training and any final scientific conclusions from it.
- Executing these notebooks inside a live Kaggle session or testing CUDA hardware.

The smoke-test results are only technical validation and are excluded from Git and the delivery ZIP.
The prepared notebooks are intended to perform the full experiment on the user's Kaggle account.
