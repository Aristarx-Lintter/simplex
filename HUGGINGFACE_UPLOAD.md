# Hugging Face Upload Guide

This guide explains how to upload the Epsilon-Transformers models and analysis data to Hugging Face.

## Prerequisites

1. **Install the dependency**:
   ```bash
   uv add huggingface_hub
   # or if dependency was already added to pyproject.toml:
   uv sync
   ```

2. **Hugging Face Authentication**:
   ```bash
   # Option 1: Login interactively
   huggingface-cli login
   
   # Option 2: Set token directly
   export HF_TOKEN="your_token_here"
   ```

3. **Create HF Repository** (optional - script will auto-create):
   Go to https://huggingface.co/new-dataset and create a dataset repo named `epsilon-transformers-belief-analysis`

## Usage

### Basic Upload (Everything)
```bash
python scripts/upload_to_huggingface.py --repo-id your-username/epsilon-transformers-belief-analysis
```

### Test Locally First
```bash
# Prepare files locally without uploading
python scripts/upload_to_huggingface.py --repo-id your-username/epsilon-transformers-belief-analysis --local-only
```

### Upload Only Analysis Files (Skip S3 Models)
```bash
python scripts/upload_to_huggingface.py --repo-id your-username/epsilon-transformers-belief-analysis --skip-models
```

### Upload Only Models (Skip Analysis)
```bash
python scripts/upload_to_huggingface.py --repo-id your-username/epsilon-transformers-belief-analysis --skip-analysis
```

## What Gets Uploaded

### Models Directory (from S3)
For each of the 16 model configurations:
- `{sweep_id}_{run_id}/`
  - `0.pt` - Initial checkpoint
  - `{final_checkpoint}.pt` - Final trained checkpoint  
  - `run_config.yaml` - Training configuration
  - `loss.csv` - Training loss data

### Analysis Directory (from local)
Exact copy of `scripts/activation_analysis/regression_analysis_files/`:
- `{sweep_id}_{run_id}/`
  - `checkpoint_*.joblib` - Regression analysis results
  - `ground_truth_data.joblib` - Neural network ground truth
  - `markov3_checkpoint_*.joblib` - Classical Markov comparisons
  - `markov3_ground_truth_data.joblib` - Classical ground truth

### Documentation
- `README.md` - Comprehensive documentation with:
  - Model mappings table
  - Process descriptions  
  - Usage examples
  - File format explanations

## Model Mappings

| Process | LSTM | GRU | RNN | Transformer |
|---------|------|-----|-----|-------------|
| **Mess3** | 20241121152808_55 | 20241121152808_63 | 20241121152808_71 | 20241205175736_23 |
| **FRDN** | 20241121152808_53 | 20241121152808_61 | 20241121152808_69 | 20250422023003_1 |
| **Bloch Walk** | 20241121152808_49 | 20241121152808_57 | 20241121152808_65 | 20241205175736_17 |
| **Moon Process** | 20241121152808_48 | 20241121152808_56 | 20241121152808_64 | 20250421221507_0 |

## Troubleshooting

### S3 Access Issues
Make sure your AWS credentials are set up for the S3ModelLoader:
```bash
# Check your S3 access
python -c "from epsilon_transformers.analysis.load_data import S3ModelLoader; s3 = S3ModelLoader(use_company_credentials=True); print(s3.list_sweeps())"
```

### Large File Upload
The script handles large files automatically. If upload fails:
- The script will resume from where it left off
- Use `--skip-models` to upload just analysis first
- Then use `--skip-analysis` to upload models separately

### Authentication Issues
```bash
# Clear cached tokens and re-authenticate
rm ~/.cache/huggingface/token
huggingface-cli login
```

## Expected Upload Size

- **Models**: ~16 model directories with checkpoints (varies by training length)
- **Analysis**: ~16 directories with regression analysis files
- **Total**: Likely several GB depending on checkpoint sizes

The upload will take some time due to the large number of files.