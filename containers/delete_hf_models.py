#!/usr/bin/env python3
"""
Script to find and delete HuggingFace model folders.
Looks for directories containing .safetensors files or other typical model files.
"""

import os
import sys
import shutil
from pathlib import Path

def is_hf_model_folder(folder_path):
    """Check if a folder looks like a HuggingFace model folder."""
    path = Path(folder_path)
    
    # Check for .safetensors files
    if list(path.glob('*.safetensors')):
        return True
    
    # Check for other model files (at least 2 indicators)
    indicator_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    found = sum(1 for f in indicator_files if (path / f).exists())
    
    return found >= 2

def find_hf_model_folders(root_dir):
    """Find all HuggingFace model folders in the directory tree."""
    model_folders = []
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        sys.exit(1)
    
    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        if is_hf_model_folder(dirpath):
            model_folders.append(dirpath)
            # Don't traverse into model folders
            dirnames.clear()

    return model_folders

def main():
    if len(sys.argv) != 2:
        print("Usage: python delete_hf_models.py <directory>")
        sys.exit(1)
    
    search_dir = sys.argv[1]
    
    print(f"Searching for HuggingFace model folders in: {search_dir}")
    model_folders = find_hf_model_folders(search_dir)
    
    if not model_folders:
        print("No HuggingFace model folders found.")
        return
    
    print(f"\nFound {len(model_folders)} model folder(s):")
    for folder in model_folders:
        print(f"  - {folder}")
    
    for folder in model_folders:
        try:
            shutil.rmtree(folder)
            print(f"Deleted: {folder}")
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
    print("\nDeletion complete!")

if __name__ == '__main__':
    main()