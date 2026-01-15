"""Download every cached Hugging Face model and dataset if missing."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Tuple

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
RESOURCES_FILE = SCRIPT_DIR / 'resources.json'

CACHE_ROOT = Path(os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface')
HUB_ROOT = CACHE_ROOT / 'hub'
MODEL_CACHE_DIRS = tuple(dict.fromkeys([
    HUB_ROOT,
    CACHE_ROOT,
    Path(os.environ.get('TRANSFORMERS_CACHE') or (CACHE_ROOT / 'models'))
]))
DATASET_CACHE_DIR = Path(os.environ.get('HF_DATASETS_CACHE') or (CACHE_ROOT / 'datasets'))

# Thread-safe printing
_print_lock = Lock()


def _safe_print(msg: str) -> None:
    """Thread-safe print."""
    with _print_lock:
        print(msg)


def _repo_folder(prefix: str, repo_id: str) -> str:
    """Build the cache folder name for a HuggingFace repo."""
    owner, name = repo_id.split('/', 1)
    return f"{prefix}--{owner}--{name}"


def _to_cache_key(dataset_name: str) -> str:
    """Convert dataset name to the cache key format used by datasets library."""
    owner, name = dataset_name.split('/', 1)
    # Convert CamelCase to snake_case and lowercase
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name).lower()
    owner = re.sub(r'([a-z])([A-Z])', r'\1_\2', owner)
    return f"{owner}___{name}"


def _any_exists(paths: List[Path]) -> bool:
    """Check if any of the given paths exist."""
    return any(p and p.exists() for p in paths)


def load_resources() -> dict:
    """Load models and datasets from resources.json."""
    with open(RESOURCES_FILE) as f:
        return json.load(f)


def _download_model(model_name: str, index: int, total: int, dry_run: bool) -> Tuple[str, bool]:
    """Download a single model. Returns (model_name, success)."""
    repo_folder = _repo_folder('models', model_name)
    candidates = [base / repo_folder for base in MODEL_CACHE_DIRS]

    if _any_exists(candidates):
        _safe_print(f"[{index}/{total}] Skipping model: {model_name} (already cached)")
        return model_name, True

    if dry_run:
        _safe_print(f"[{index}/{total}] Would download model: {model_name}")
        return model_name, True

    _safe_print(f"[{index}/{total}] Downloading model: {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    _safe_print(f"[{index}/{total}] Model {model_name} downloaded successfully")
    return model_name, True


def _download_dataset(entry: dict, index: int, total: int, dry_run: bool) -> Tuple[str, bool]:
    """Download a single dataset. Returns (dataset_name, success)."""
    dataset_name = entry['dataset']
    configs = entry.get('configs', [entry.get('config', 'default')])
    splits = entry.get('splits', [])

    # Check if already cached
    repo_folder = _repo_folder('datasets', dataset_name)
    cache_key = _to_cache_key(dataset_name)
    cached = _any_exists([
        HUB_ROOT / repo_folder,
        CACHE_ROOT / repo_folder,
        DATASET_CACHE_DIR / cache_key
    ])

    if cached:
        _safe_print(f"[{index}/{total}] Skipping dataset: {dataset_name} (already cached)")
        return dataset_name, True

    # Download each config
    for config in configs:
        label = f"{dataset_name} ({config})" if config else dataset_name

        if dry_run:
            if splits:
                _safe_print(f"[{index}/{total}] Would download dataset: {label} [splits={splits}]")
            else:
                _safe_print(f"[{index}/{total}] Would download dataset: {label}")
            continue

        if splits:
            for split in splits:
                _safe_print(f"[{index}/{total}] Downloading dataset: {label} [split={split}]...")
                kwargs = {'split': split}
                if config and config != 'default':
                    kwargs['name'] = config
                load_dataset(dataset_name, **kwargs)
        else:
            _safe_print(f"[{index}/{total}] Downloading dataset: {label}...")
            kwargs = {}
            if config and config != 'default':
                kwargs['name'] = config
            load_dataset(dataset_name, **kwargs)

    if not dry_run:
        _safe_print(f"[{index}/{total}] Dataset {dataset_name} downloaded successfully")
    return dataset_name, True


def download_models(models: List[str], dry_run: bool = False) -> None:
    """Download all models that aren't already cached."""
    total = len(models)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(_download_model, model, i, total, dry_run): model
            for i, model in enumerate(models, 1)
        }
        for future in as_completed(futures):
            future.result()  # Raise any exceptions


def download_datasets(datasets: List[dict], dry_run: bool = False, workers: int = 4) -> None:
    """Download all datasets that aren't already cached."""
    total = len(datasets)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_dataset, entry, i, total, dry_run): entry['dataset']
            for i, entry in enumerate(datasets, 1)
        }
        for future in as_completed(futures):
            future.result()  # Raise any exceptions


def main(dry_run: bool = False, workers: int = 4) -> None:
    """Main entry point."""
    resources = load_resources()

    print(f"Models: {len(resources['models'])}")
    print(f"Datasets: {len(resources['datasets'])}")
    print(f"Workers: {workers}")
    if dry_run:
        print("DRY RUN - no downloads will be performed")
    print()

    download_models(resources['models'], dry_run=dry_run)
    print()
    download_datasets(resources['datasets'], dry_run=dry_run, workers=workers)

    print(f"\nCache location: {CACHE_ROOT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download HuggingFace models and datasets')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without actually downloading')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel download workers (default: 4)')
    args = parser.parse_args()

    main(dry_run=args.dry_run, workers=args.workers)
