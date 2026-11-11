"""Create the two importable Kaggle notebooks; no credentials are embedded."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}

def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.splitlines(keepends=True)}

SETUP = '''from pathlib import Path
import subprocess, sys, shutil

REPO = Path('/kaggle/working/TTKHTKUD_26-27')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/trungnd06/TTKHTKUD_26-27.git', str(REPO)], check=True)
SOURCE = REPO / 'kaggle'
assert (SOURCE / 'prepare_data.py').is_file(), 'This clone does not contain kaggle/. Use the updated repository.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(SOURCE / 'requirements.txt')], check=True)
print('Code:', SOURCE)
'''

def save(name, cells, gpu=False):
    data = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': gpu}},
        'nbformat': 4, 'nbformat_minor': 5}
    # Stable cell IDs support current nbformat validators.
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:02d}'
    (ROOT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

save('01_prepare_full.ipynb', [
    md('# 1. Prepare all Nencki-Symfonia oddball subjects\n'
       'Enable Internet; CPU is sufficient. Run all 42 subjects, save the output version, then attach it to notebook 02. '
       'Raw data is downloaded one subject at a time and cleaned up. Signal-quality exclusions are logged, never hidden.'),
    code(SETUP),
    code('''# Optional: an attached BIDS raw dataset root. None = download from public OpenNeuro.
RAW_ROOT = None  # Example: '/kaggle/input/my-nencki-raw/ds004621'
# Optional: previously saved processed folder, to resume completed subjects.
RESUME_PROCESSED = None
OUTPUT = Path('/kaggle/working/nencki_full/processed')
SUBJECTS = list(range(1, 43))  # Keep all 42 for the full experiment.

if RESUME_PROCESSED is not None:
    source = Path(RESUME_PROCESSED).resolve()
    assert source.is_dir() and source != OUTPUT.resolve()
    shutil.copytree(source, OUTPUT, dirs_exist_ok=True)
command = [sys.executable, str(SOURCE / 'prepare_data.py'), '--output', str(OUTPUT),
           '--subjects', *map(str, SUBJECTS)]
if RAW_ROOT is not None:
    command += ['--raw-root', str(RAW_ROOT)]
subprocess.run(command, check=True)
'''),
    code('''import json
summary = json.loads((OUTPUT / 'preparation_summary.json').read_text())
print('All 42 attempted:', summary['all_42_attempted'])
print('Retained subjects:', len(summary['included_subjects']))
print('Quality exclusions:', summary['excluded_subjects'])
print('Standard / target:', summary['n_standard'], summary['n_target'])
print('Shape:', summary['shape'])
print('Output MiB:', round(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()) / 1024**2, 1))
'''),
    md('## Save the preparation output\nChoose **Save Version** and retain output files. '
       'In notebook 02, Add Input and attach this output. Alternatively download the whole processed folder '
       'and create your own Kaggle Dataset. Keep `preparation_summary.json` beside `epochs_full.npz`. '
       'The script can resume subjects only from files actually retained or restored in a later session.')
])

save('02_train_full.ipynb', [
    md('# 2. Train and explain the full-cohort EEG models\n'
       'Attach the saved output of notebook 01 using Add Input. Enable a GPU and Internet. '
       'The CNN uses one CUDA device; classical methods use CPU. This notebook does not use the eight-subject pilot by default.'),
    code(SETUP),
    code('''import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    raise RuntimeError('Enable a Kaggle GPU before training, or deliberately select DEVICE=cpu and remove this guard.')
'''),
    code('''# Fill DATA_PATH if automatic discovery finds more than one file.
DATA_PATH = None  # Example: '/kaggle/input/my-full-eeg/processed/epochs_full.npz'
RESUME_RESULTS = None  # Optional folder of previously saved results, including cached_folds/ and models/.
RESULTS = Path('/kaggle/working/nencki_full/results')
DEVICE = 'cuda'
EPOCHS = 50
FOLDS = 5
SEED = 42

if DATA_PATH is None:
    candidates = sorted(Path('/kaggle/input').rglob('epochs_full.npz'))
    local = Path('/kaggle/working/nencki_full/processed/epochs_full.npz')
    if local.is_file():
        candidates.append(local)
    if len(candidates) != 1:
        raise RuntimeError(f'Found {len(candidates)} prepared datasets. Attach notebook 01 output or set DATA_PATH explicitly. {candidates}')
    DATA_PATH = candidates[0]
DATA_PATH = Path(DATA_PATH)
assert (DATA_PATH.parent / 'preparation_summary.json').is_file()
if RESUME_RESULTS is not None:
    source = Path(RESUME_RESULTS).resolve()
    assert source.is_dir() and source != RESULTS.resolve()
    shutil.copytree(source, RESULTS, dirs_exist_ok=True)
print('Data:', DATA_PATH)
'''),
    code('''subprocess.run([sys.executable, str(SOURCE / 'train.py'), '--data', str(DATA_PATH),
                '--output', str(RESULTS), '--device', DEVICE, '--epochs', str(EPOCHS),
                '--folds', str(FOLDS), '--seed', str(SEED)], check=True)
'''),
    code('''subprocess.run([sys.executable, str(SOURCE / 'verify.py'), '--data', str(DATA_PATH),
                '--output', str(RESULTS)], check=True)
import pandas as pd
display(pd.read_csv(RESULTS / 'summary.csv', header=[0, 1], index_col=0))
print((RESULTS / 'RESULTS_EN.md').read_text())
'''),
    code('''from IPython.display import Image, display
for name in ['comparison.png', 'erp.png', 'attribution.png', 'occlusion.png']:
    display(Image(filename=str(RESULTS / name)))
'''),
    code('''# Save Version with outputs, or download this results archive.
archive = shutil.make_archive('/kaggle/working/nencki_full_results', 'zip', root_dir=RESULTS)
print(archive)
'''),
    md('## Interpret the measured results\nUse the actual retained subject count and QC exclusions in the report. '
       'One base seed and fold SD do not establish statistical or universal superiority. '
       'Attributions are model sensitivity, not neural-source localization. '
       'The parent project report remains the eight-subject pilot until the group updates it using these measured results.')
], gpu=True)
print('Created 01_prepare_full.ipynb and 02_train_full.ipynb')
