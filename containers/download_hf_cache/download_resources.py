"""Download every cached Hugging Face model and dataset if missing."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional

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


def download_models(models: List[str], dry_run: bool = False) -> None:
    """Download all models that aren't already cached."""
    total = len(models)
    for i, model_name in enumerate(models, 1):
        repo_folder = _repo_folder('models', model_name)
        candidates = [base / repo_folder for base in MODEL_CACHE_DIRS]

        if _any_exists(candidates):
            print(f"[{i}/{total}] Skipping model: {model_name} (already cached)")
            continue

        if dry_run:
            print(f"[{i}/{total}] Would download model: {model_name}")
            continue

        print(f"[{i}/{total}] Downloading model: {model_name}...")
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        print(f"[{i}/{total}] Model {model_name} downloaded successfully")


def download_datasets(datasets: List[dict], dry_run: bool = False) -> None:
    """Download all datasets that aren't already cached."""
    total = len(datasets)
    for i, entry in enumerate(datasets, 1):
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
            print(f"[{i}/{total}] Skipping dataset: {dataset_name} (already cached)")
            continue

        # Download each config
        for config in configs:
            label = f"{dataset_name} ({config})" if config else dataset_name

            if dry_run:
                if splits:
                    print(f"[{i}/{total}] Would download dataset: {label} [splits={splits}]")
                else:
                    print(f"[{i}/{total}] Would download dataset: {label}")
                continue

            if splits:
                for split in splits:
                    print(f"[{i}/{total}] Downloading dataset: {label} [split={split}]...")
                    kwargs = {'split': split}
                    if config and config != 'default':
                        kwargs['name'] = config
                    load_dataset(dataset_name, **kwargs)
            else:
                print(f"[{i}/{total}] Downloading dataset: {label}...")
                kwargs = {}
                if config and config != 'default':
                    kwargs['name'] = config
                load_dataset(dataset_name, **kwargs)

        if not dry_run:
            print(f"[{i}/{total}] Dataset {dataset_name} downloaded successfully")


def main(dry_run: bool = False) -> None:
    """Main entry point."""
    resources = load_resources()

    print(f"Models: {len(resources['models'])}")
    print(f"Datasets: {len(resources['datasets'])}")
    if dry_run:
        print("DRY RUN - no downloads will be performed\n")
    print()

    download_models(resources['models'], dry_run=dry_run)
    print()
    download_datasets(resources['datasets'], dry_run=dry_run)

    print(f"\nCache location: {CACHE_ROOT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download HuggingFace models and datasets')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without actually downloading')
    args = parser.parse_args()

    main(dry_run=args.dry_run)
