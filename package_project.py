"""Create a compact hand-in archive (exclude multi-GB raw files and local libraries)."""
import bootstrap
import json
from pathlib import Path
import zipfile

def main():
    root = Path(__file__).resolve().parent
    dest = root / 'Miniproject_EEG_Group.zip'
    files = []
    for pattern in ['*.py', 'requirements*.txt', 'README.md', '.gitignore', '.gitattributes',
                    'deliverables/*', 'results/*', 'results/models/*',
                    'data/epochs.npz', 'data/*.json', 'data/README.md', 'docs/*', 'kaggle/*',
                    'deliverables_full/*', 'results_full/*', 'results_full/models/*',
                    'results_full/cached_folds/*']:
        files.extend(p for p in root.glob(pattern) if p.is_file())
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        entries = archive.namelist()
        assert 'deliverables/Report_EN.pdf' in entries and 'data/epochs.npz' in entries
        assert not any(n.startswith(('.vendor/', 'data/raw/')) for n in entries)
    print(f'{dest.name}: {dest.stat().st_size / 1024**2:.1f} MiB, {len(entries)} files; integrity OK')

if __name__ == '__main__':
    main()
