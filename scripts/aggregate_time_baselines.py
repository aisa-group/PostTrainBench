import os
import csv
from pathlib import Path
from collections import defaultdict

def parse_directory_name(dirname):
    """Extract benchmark and model from directory name."""
    parts = dirname.split('_')

    benchmarks = []
    benchmarks_path = Path('src/eval/tasks')
    for item in benchmarks_path.iterdir():
        benchmarks.append(item.name)

    for benchmark in benchmarks:
        prefix = benchmark + '_'
        if dirname.startswith(prefix):
            rest = dirname[len(prefix):]
            break
    else:
        return None, None, None
    
    # Extract model name and run ID
    parts = rest.rsplit('_', 1)
    if len(parts) == 2:
        model = parts[0].replace('_', '/')  # Convert back to org/model format
        run_id = parts[1]
        return benchmark, model, run_id
    
    return None, None, None

def get_latest_results(results_dir):
    """Get the latest run for each model-benchmark combination."""
    results_path = Path(results_dir)
    
    # Store all runs grouped by (benchmark, model)
    runs = defaultdict(list)
    
    # Scan all directories
    for subdir in results_path.glob('baseline_zeroshot/*'):
        if subdir.is_dir():
            benchmark, model, run_id = parse_directory_name(subdir.name)
            
            if benchmark and model and run_id:
                time_file = subdir / 'time_taken.txt'
                runs[(benchmark, model)].append({
                    'path': time_file,
                    'run_id': run_id,
                    'dir_name': subdir.name,
                    'exists': time_file.exists()
                })
    
    # Get the latest run for each combination (highest run_id)
    latest_runs = {}
    for (benchmark, model), run_list in runs.items():
        latest = max(run_list, key=lambda x: x['run_id'])
        latest_runs[(benchmark, model)] = {
            'path': latest['path'],
            'exists': latest['exists']
        }
    
    return latest_runs

def read_time_taken(time_path):
    """Read time_taken.txt and return the time string, removing leading zeros."""
    with open(time_path, 'r') as f:
        time_str = f.read().strip()
    
    # If format is HH:MM:SS and hours are 00, strip to MM:SS
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3 and parts[0] == '00':
            time_str = f"{parts[1]}:{parts[2]}"
    
    return time_str

def create_results_csv(results_dir, output_file='benchmark_times.csv'):
    """Create a CSV file with aggregated time taken results."""
    # Get latest results
    latest_runs = get_latest_results(results_dir)
    
    # Collect all unique benchmarks and models
    benchmarks = sorted(set(b for b, m in latest_runs.keys()))
    models = sorted(set(m for b, m in latest_runs.keys()))
    
    # Create results matrix
    results = {}
    for model in models:
        results[model] = {}
        for benchmark in benchmarks:
            if (benchmark, model) in latest_runs:
                run_info = latest_runs[(benchmark, model)]
                if run_info['exists']:
                    results[model][benchmark] = read_time_taken(str(run_info['path']))
                else:
                    results[model][benchmark] = 'ERR'
            else:
                results[model][benchmark] = 'N/A'
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow(['Model'] + benchmarks)
        
        # Data rows - prefix times with single quote to force string interpretation
        for model in models:
            row = [model] + [f"'{results[model][b]}" for b in benchmarks]
            writer.writerow(row)
    
    print(f"Results saved to {output_file}")
    print(f"Models: {len(models)}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Total entries: {len(latest_runs)}")
    
    # Count ERR entries
    err_count = sum(1 for model in models for b in benchmarks if results[model][b] == 'ERR')
    if err_count > 0:
        print(f"Warning: {err_count} entries missing time_taken.txt (marked as ERR)")

def get_results_dir():
    return os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", 'results')

if __name__ == '__main__':
    results_dir = get_results_dir()
    
    create_results_csv(results_dir, os.path.join(results_dir, 'benchmark_times.csv'))