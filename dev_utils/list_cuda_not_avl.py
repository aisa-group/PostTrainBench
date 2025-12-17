#!/usr/bin/env python3
import os
from pathlib import Path

def check_cuda_availability():
    results_dir = os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')

    base_path = Path(results_dir)
    
    # Iterate through subdirs
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Iterate through subsubdirs
        for subsubdir in subdir.iterdir():
            if not subsubdir.is_dir():
                continue
                
            # Check if task/cuda_not_available exists
            cuda_not_available = subsubdir / "task" / "cuda_not_available"
            if cuda_not_available.exists():
                print(subsubdir)

if __name__ == "__main__":
    check_cuda_availability()