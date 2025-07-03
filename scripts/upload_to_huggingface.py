#!/usr/bin/env python3
"""
Upload Epsilon-Transformers models and analysis to Hugging Face

This script uploads both:
1. Model checkpoints from S3 to HF models/ directory
2. Local regression analysis files to HF analysis/ directory

Usage:
    python scripts/upload_to_huggingface.py --repo-id your-username/epsilon-transformers-belief-analysis
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import tempfile

from huggingface_hub import HfApi, login, create_repo
from epsilon_transformers.analysis.load_data import S3ModelLoader

# Model mappings: (sweep_id, run_id) -> (architecture, process)
MODEL_MAPPINGS = {
    # Mess3 (Classical Process)
    ("20241121152808", "55"): ("LSTM", "Mess3"),
    ("20241205175736", "23"): ("Transformer", "Mess3"),
    ("20241121152808", "63"): ("GRU", "Mess3"),
    ("20241121152808", "71"): ("RNN", "Mess3"),
    
    # FRDN (Quantum Process)
    ("20241121152808", "53"): ("LSTM", "FRDN"),
    ("20250422023003", "1"): ("Transformer", "FRDN"),
    ("20241121152808", "61"): ("GRU", "FRDN"),
    ("20241121152808", "69"): ("RNN", "FRDN"),
    
    # Bloch Walk (Quantum Process)
    ("20241121152808", "49"): ("LSTM", "Bloch Walk"),
    ("20241205175736", "17"): ("Transformer", "Bloch Walk"),
    ("20241121152808", "57"): ("GRU", "Bloch Walk"),
    ("20241121152808", "65"): ("RNN", "Bloch Walk"),
    
    # Moon Process (Post-Quantum Process)
    ("20241121152808", "48"): ("LSTM", "Moon Process"),
    ("20250421221507", "0"): ("Transformer", "Moon Process"),
    ("20241121152808", "56"): ("GRU", "Moon Process"),
    ("20241121152808", "64"): ("RNN", "Moon Process"),
}

def get_all_sweep_run_pairs() -> List[Tuple[str, str]]:
    """Get all sweep/run pairs from MODEL_MAPPINGS"""
    return list(MODEL_MAPPINGS.keys())

def download_model_from_s3(sweep_id: str, run_id: str, local_dir: Path, s3_loader: S3ModelLoader):
    """Download a single model's files from S3 to local directory"""
    print(f"Downloading model {sweep_id}_{run_id} from S3...")
    
    # Create local directory
    model_dir = local_dir / f"{sweep_id}_{run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get runs in sweep and find matching run
        runs = s3_loader.list_runs_in_sweep(sweep_id)
        matching_run = None
        for run in runs:
            if f"run_{run_id}" in run:
                matching_run = run
                break
        
        if not matching_run:
            print(f"ERROR: Could not find run_{run_id} in sweep {sweep_id}")
            return False
        
        # Download checkpoints using S3 client directly
        checkpoints = s3_loader.list_checkpoints(sweep_id, matching_run)
        print(f"Found {len(checkpoints)} checkpoints for {sweep_id}_{run_id}")
        
        # Download all checkpoint files using S3 client
        for checkpoint in checkpoints:
            checkpoint_name = checkpoint.split('/')[-1]
            local_path = model_dir / checkpoint_name
            
            try:
                # Add path prefix to checkpoint path for S3
                full_s3_key = f"{s3_loader.path_prefix}{checkpoint}"
                
                # Use S3 client to download file
                s3_loader.s3_client.download_file(
                    s3_loader.bucket_name, 
                    full_s3_key, 
                    str(local_path)
                )
                print(f"  Downloaded {checkpoint_name}")
            except Exception as e:
                print(f"  Warning: Could not download {checkpoint_name}: {e}")
        
        # Download config files using S3 client
        try:
            # Try to find run_config.yaml
            config_key = f"{s3_loader.path_prefix}{sweep_id}/{matching_run}/run_config.yaml"
            config_path = model_dir / "run_config.yaml"
            s3_loader.s3_client.download_file(
                s3_loader.bucket_name,
                config_key,
                str(config_path)
            )
            print(f"  Downloaded run_config.yaml")
        except Exception as e:
            print(f"  Warning: Could not download run_config.yaml: {e}")
        
        # Download logs
        try:
            loss_df = s3_loader.load_loss_from_run(sweep_id, matching_run)
            loss_path = model_dir / "loss.csv"
            loss_df.to_csv(loss_path, index=False)
            print(f"  Downloaded loss.csv")
        except Exception as e:
            print(f"  Warning: Could not download loss data: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR downloading {sweep_id}_{run_id}: {e}")
        return False

def copy_analysis_files(analysis_source_dir: Path, local_dir: Path):
    """Copy analysis files from local regression_analysis_files to upload directory"""
    print("Copying analysis files...")
    
    analysis_dir = local_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy each sweep_run directory
    for sweep_id, run_id in get_all_sweep_run_pairs():
        source_folder = analysis_source_dir / f"{sweep_id}_{run_id}"
        dest_folder = analysis_dir / f"{sweep_id}_{run_id}"
        
        if source_folder.exists():
            shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
            print(f"  Copied {sweep_id}_{run_id} analysis files")
        else:
            print(f"  WARNING: Analysis files not found for {sweep_id}_{run_id}")

def generate_readme(local_dir: Path):
    """Generate README.md with model mappings and documentation"""
    readme_content = """# Epsilon-Transformers Belief Analysis Dataset

This dataset contains trained neural network models and their corresponding belief state regression analysis from the Epsilon-Transformers project. The models were trained on four different stochastic processes and analyzed for their ability to learn and represent belief states.

## Dataset Structure

```
epsilon-transformers-belief-analysis/
├── README.md
├── models/          # Model checkpoints and configurations from S3
│   ├── {sweep_id}_{run_id}/
│   │   ├── 0.pt                    # Initial checkpoint
│   │   ├── {final}.pt              # Final checkpoint
│   │   ├── run_config.yaml         # Training configuration
│   │   └── loss.csv                # Training loss data
│   └── ...
└── analysis/        # Belief state regression analysis results
    ├── {sweep_id}_{run_id}/
    │   ├── checkpoint_0.joblib              # Initial checkpoint analysis
    │   ├── checkpoint_{final}.joblib        # Final checkpoint analysis
    │   ├── ground_truth_data.joblib         # Neural network ground truth
    │   ├── markov3_checkpoint_*.joblib      # Classical Markov comparisons
    │   └── markov3_ground_truth_data.joblib # Classical ground truth
    └── ...
```

## Model Mappings

| Sweep ID | Run ID | Architecture | Process | Description |
|----------|--------|--------------|---------|-------------|
"""
    
    # Add model mappings table
    processes = ["Mess3", "FRDN", "Bloch Walk", "Moon Process"]
    architectures = ["LSTM", "GRU", "RNN", "Transformer"]
    
    for process in processes:
        readme_content += f"\n### {process}\n"
        for arch in architectures:
            for (sweep_id, run_id), (model_arch, model_process) in MODEL_MAPPINGS.items():
                if model_arch == arch and model_process == process:
                    readme_content += f"| {sweep_id} | {run_id} | {arch} | {process} | {arch} trained on {process} |\n"
    
    readme_content += """

## Process Descriptions

### Mess3 (Classical Process)
A classical stochastic process used as a baseline for comparison with quantum processes.

### FRDN (Finite Random Dynamics Networks)
A quantum process representing finite random dynamics networks, modeling quantum systems with specific structural properties.

### Bloch Walk
A quantum random walk process on the Bloch sphere, representing quantum state evolution in a geometric framework.

### Moon Process
A post-quantum stochastic process that explores computational mechanics beyond standard quantum frameworks.

## Model Architectures

### RNN Models (LSTM, GRU, RNN)
- **Layers**: 4
- **Hidden Units**: 64
- **Direction**: Unidirectional
- **Configuration**: L4_H64_uni

### Transformer Models
- **Layers**: 4
- **Attention Heads**: 4
- **Head Dimension**: 16
- **Model Dimension**: 64
- **Configuration**: L4_H4_DH16_DM64

## File Formats

### Model Files (.pt)
PyTorch model checkpoints containing trained model weights and optimizer states.

### Analysis Files (.joblib)
Joblib-serialized files containing:
- **checkpoint_*.joblib**: Regression analysis results mapping activations to belief states
- **ground_truth_data.joblib**: True belief states and probabilities for the neural network data
- **markov3_*.joblib**: Classical Markov model comparisons and baselines

## Usage

### Loading Models
```python
import torch
from pathlib import Path

# Load a model checkpoint
model_path = Path("models/20241121152808_57/4075724800.pt")
checkpoint = torch.load(model_path, map_location='cpu')
```

### Loading Analysis Data
```python
import joblib
from pathlib import Path

# Load regression analysis results
analysis_path = Path("analysis/20241121152808_57/checkpoint_4075724800.joblib")
analysis_data = joblib.load(analysis_path)

# Access layer-wise regression metrics
for layer, metrics in analysis_data.items():
    print(f"Layer {layer} RMSE: {metrics['rmse']}")
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{epsilon-transformers-belief-analysis,
  title={Epsilon-Transformers Belief Analysis Dataset},
  author={[Your Name]},
  year={2024},
  howpublished={Hugging Face Datasets},
  url={https://huggingface.co/datasets/[your-username]/epsilon-transformers-belief-analysis}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]
"""
    
    readme_path = local_dir / "README.md"
    readme_path.write_text(readme_content)
    print(f"Generated README.md")

def upload_to_huggingface(repo_id: str, local_dir: Path, token: str = None):
    """Upload the dataset to Hugging Face"""
    print(f"Uploading to Hugging Face repository: {repo_id}")
    
    # Login to HF only if token is explicitly provided
    if token:
        login(token=token)
    # Otherwise use existing cached token (don't call login again)
    
    api = HfApi(token=token)  # Use token if provided, otherwise use cached
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Note: Repository may already exist: {e}")
    
    # Upload the entire directory
    print("Starting upload...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload epsilon-transformers models and belief analysis data"
    )
    print(f"Upload complete! Dataset available at: https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload Epsilon-Transformers to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repository ID (username/repo-name)")
    parser.add_argument("--token", help="HuggingFace API token (optional if cached)")
    parser.add_argument("--analysis-dir", default="scripts/activation_analysis/regression_analysis_files", 
                       help="Path to local analysis files")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models from S3")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip copying analysis files")
    parser.add_argument("--local-only", action="store_true", help="Prepare files locally but don't upload")
    parser.add_argument("--streaming", action="store_true", help="Upload models one at a time to save disk space")
    
    args = parser.parse_args()
    
    # Set up paths
    current_dir = Path.cwd()
    analysis_source_dir = current_dir / args.analysis_dir
    
    print(f"Starting upload preparation for {args.repo_id}")
    print(f"Analysis source: {analysis_source_dir}")
    print(f"Streaming mode: {'ON' if args.streaming else 'OFF'}")
    
    # Initialize HF API once
    api = HfApi(token=args.token)
    if not args.local_only:
        # Create repository if it doesn't exist
        try:
            create_repo(args.repo_id, repo_type="dataset", exist_ok=True, token=args.token)
            print(f"Repository {args.repo_id} created/verified")
        except Exception as e:
            print(f"Note: Repository may already exist: {e}")
    
    if args.streaming and not args.skip_models and not args.local_only:
        # STREAMING MODE: Upload models one at a time
        print("\n=== Streaming Upload Mode ===")
        s3_loader = S3ModelLoader(use_company_credentials=True)
        
        successful_uploads = 0
        for i, (sweep_id, run_id) in enumerate(get_all_sweep_run_pairs()):
            print(f"\n--- Processing model {i+1}/{len(MODEL_MAPPINGS)}: {sweep_id}_{run_id} ---")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                local_dir = Path(temp_dir)
                models_dir = local_dir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Download single model
                if download_model_from_s3(sweep_id, run_id, models_dir, s3_loader):
                    # Upload just this model
                    try:
                        model_folder = models_dir / f"{sweep_id}_{run_id}"
                        api.upload_folder(
                            folder_path=str(model_folder),
                            repo_id=args.repo_id,
                            repo_type="dataset",
                            path_in_repo=f"models/{sweep_id}_{run_id}",
                            commit_message=f"Upload model {sweep_id}_{run_id}"
                        )
                        print(f"  ✅ Uploaded {sweep_id}_{run_id}")
                        successful_uploads += 1
                    except Exception as e:
                        print(f"  ❌ Failed to upload {sweep_id}_{run_id}: {e}")
                else:
                    print(f"  ❌ Failed to download {sweep_id}_{run_id}")
                
                # Temp directory auto-deletes here
        
        print(f"\nStreaming upload complete: {successful_uploads}/{len(MODEL_MAPPINGS)} models uploaded")
        
        # Upload analysis files and README separately
        if not args.skip_analysis:
            print("\n=== Uploading Analysis Files ===")
            with tempfile.TemporaryDirectory() as temp_dir:
                local_dir = Path(temp_dir)
                copy_analysis_files(analysis_source_dir, local_dir)
                generate_readme(local_dir)
                
                api.upload_folder(
                    folder_path=str(local_dir),
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message="Upload analysis files and README"
                )
                print("✅ Analysis files and README uploaded")
    
    else:
        # ORIGINAL MODE: Download all, then upload all
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir) / "upload"
            local_dir.mkdir(parents=True)
            
            # Download models from S3
            if not args.skip_models:
                print("\n=== Downloading Models from S3 ===")
                models_dir = local_dir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                
                s3_loader = S3ModelLoader(use_company_credentials=True)
                
                successful_downloads = 0
                for sweep_id, run_id in get_all_sweep_run_pairs():
                    if download_model_from_s3(sweep_id, run_id, models_dir, s3_loader):
                        successful_downloads += 1
                
                print(f"Successfully downloaded {successful_downloads}/{len(MODEL_MAPPINGS)} models")
            
            # Copy analysis files
            if not args.skip_analysis:
                print("\n=== Copying Analysis Files ===")
                copy_analysis_files(analysis_source_dir, local_dir)
            
            # Generate README
            print("\n=== Generating README ===")
            generate_readme(local_dir)
            
            # Upload to HuggingFace
            if not args.local_only:
                print("\n=== Uploading to Hugging Face ===")
                upload_to_huggingface(args.repo_id, local_dir, args.token)
            else:
                print(f"\n=== Local preparation complete ===")
                print(f"Files prepared in: {local_dir}")
                print("Use --local-only=false to upload to HuggingFace")

if __name__ == "__main__":
    main()