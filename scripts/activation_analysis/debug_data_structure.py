#!/usr/bin/env python3
"""
Debug script to examine the actual data structure and available files
for all experiments and model types.
"""

import os
import glob

def debug_data_structure():
    """Debug what files and directories actually exist in our data."""
    
    base_dir = "run_predictions_RCOND_FINAL"
    
    print("=== DEBUG: Data Structure Analysis ===\n")
    
    # Define the mappings based on what you told me
    model_configs = {
        'RNN': [71, 70, 69, 65, 68, 64],      # mess3, rrxor, fanizza, tom_quantum_a, tom_quantum_b, post_quantum
        'GRU': [63, 62, 61, 57, 60, 56],      # same order
        'LSTM': [55, 54, 53, 49, 52, 48],     # same order
        'Transformer': [23, 22, 21, 17, 20, 16]  # same order
    }
    
    experiment_names = ['Mess3', 'RRXor', 'Fanizza', 'TomQA', 'TomQB', 'PostQuantum']
    
    # Check all possible run directories
    print("1. Checking which run directories exist:")
    print("-" * 50)
    
    all_dirs = glob.glob(os.path.join(base_dir, "*"))
    available_dirs = [os.path.basename(d) for d in all_dirs if os.path.isdir(d)]
    available_dirs.sort()
    
    print(f"Available directories in {base_dir}:")
    for d in available_dirs:
        print(f"  {d}")
    
    print(f"\nTotal directories found: {len(available_dirs)}")
    
    # Check for each model type and experiment
    print("\n2. Checking model run availability:")
    print("-" * 50)
    
    for model_type, run_ids in model_configs.items():
        print(f"\n{model_type} Model:")
        
        # Determine folder
        if model_type == 'Transformer':
            folder = "20241205175736"
        else:
            folder = "20241121152808"
        
        for i, run_id in enumerate(run_ids):
            exp_name = experiment_names[i]
            dir_name = f"{folder}_{run_id}"
            dir_path = os.path.join(base_dir, dir_name)
            
            exists = "✓" if os.path.exists(dir_path) else "✗"
            print(f"  {exp_name:12} ({run_id:2}): {dir_name:20} {exists}")
    
    # Check ground truth files for each experiment
    print("\n3. Checking ground truth files:")
    print("-" * 50)
    
    # Define ground truth runs (these should be the Transformer runs for ground truth)
    gt_runs = {
        'Mess3': ("20241205175736", 23),
        'RRXor': ("20241205175736", 22),
        'Fanizza': ("20241205175736", 21),
        'TomQA': ("20241205175736", 17),
        'TomQB': ("20241205175736", 20),
        'PostQuantum': ("20241205175736", 16),
    }
    
    for exp_name, (sweep, run_id) in gt_runs.items():
        gt_dir = os.path.join(base_dir, f"{sweep}_{run_id}")
        print(f"\n{exp_name} Ground Truth ({sweep}_{run_id}):")
        
        if os.path.exists(gt_dir):
            # Check for both types of ground truth files
            standard_gt = os.path.join(gt_dir, "ground_truth_data.joblib")
            markov3_gt = os.path.join(gt_dir, "markov3_ground_truth_data.joblib")
            
            std_exists = "✓" if os.path.exists(standard_gt) else "✗"
            m3_exists = "✓" if os.path.exists(markov3_gt) else "✗"
            
            print(f"  Directory exists: ✓")
            print(f"  ground_truth_data.joblib: {std_exists}")
            print(f"  markov3_ground_truth_data.joblib: {m3_exists}")
            
            # List all files in the directory
            all_files = os.listdir(gt_dir)
            joblib_files = [f for f in all_files if f.endswith('.joblib')]
            print(f"  All .joblib files ({len(joblib_files)}):")
            for f in sorted(joblib_files):
                print(f"    {f}")
        else:
            print(f"  Directory exists: ✗")
    
    # Check checkpoint files for working directories
    print("\n4. Sample checkpoint files (first few directories):")
    print("-" * 50)
    
    sample_dirs = available_dirs[:3] if len(available_dirs) >= 3 else available_dirs
    for dir_name in sample_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        print(f"\n{dir_name}:")
        
        # Look for checkpoint files
        checkpoint_files = glob.glob(os.path.join(dir_path, "checkpoint_*.joblib"))
        markov3_checkpoint_files = glob.glob(os.path.join(dir_path, "markov3_checkpoint_*.joblib"))
        
        print(f"  Regular checkpoints: {len(checkpoint_files)}")
        if checkpoint_files:
            # Show first and last checkpoint
            checkpoint_nums = []
            for f in checkpoint_files:
                try:
                    num = int(os.path.basename(f).split('_')[1].split('.')[0])
                    checkpoint_nums.append(num)
                except:
                    pass
            if checkpoint_nums:
                checkpoint_nums.sort()
                print(f"    Range: {min(checkpoint_nums)} to {max(checkpoint_nums)}")
        
        print(f"  Markov3 checkpoints: {len(markov3_checkpoint_files)}")
        if markov3_checkpoint_files:
            # Show first and last checkpoint
            checkpoint_nums = []
            for f in markov3_checkpoint_files:
                try:
                    num = int(os.path.basename(f).split('_')[2].split('.')[0])
                    checkpoint_nums.append(num)
                except:
                    pass
            if checkpoint_nums:
                checkpoint_nums.sort()
                print(f"    Range: {min(checkpoint_nums)} to {max(checkpoint_nums)}")

if __name__ == "__main__":
    debug_data_structure()