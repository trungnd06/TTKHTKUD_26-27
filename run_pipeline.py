"""Run a reproducible workflow using the current Python environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['verify', 'train', 'full'], default='verify',
                        help='verify saved results, retrain from included epochs, or rebuild from raw data')
    args = parser.parse_args()
    steps = {
        'verify': ['check_data.py', 'verify_results.py'],
        'train': ['check_data.py', 'experiment.py', 'verify_results.py'],
        'full': ['download_data.py', 'preprocess.py', 'experiment.py', 'verify_results.py'],
    }[args.mode]
    for script in steps:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print('Workflow completed. Reports and contribution files were not regenerated.', flush=True)

if __name__ == '__main__':
    main()
