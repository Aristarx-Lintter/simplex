# Epsilon-Transformers: Methods and Reproducibility Guide

This document provides comprehensive methods for reproducing the results in "Neural networks leverage nominally quantum and post-quantum representations" and accessing the associated datasets and models.

## Dataset and Model Access

### Public Dataset
The complete dataset including trained models and analysis results is publicly available on HuggingFace:

**Repository**: [`SimplexAI/quantum-representations`](https://huggingface.co/datasets/SimplexAI/quantum-representations)

The dataset contains:
- **16 trained neural network models** (4 architectures × 4 processes)
- **Complete belief state regression analysis** with k-fold cross-validation results
- **Model checkpoints** at multiple training stages
- **Training configurations and loss curves**

### Dataset Structure

```
SimplexAI/quantum-representations/
├── models/                    # Trained model checkpoints
│   ├── {sweep_id}_{run_id}/   # Model directory
│   │   ├── 0.pt              # Initial checkpoint
│   │   ├── {final}.pt        # Final trained checkpoint
│   │   ├── run_config.yaml   # Training configuration
│   │   └── loss.csv          # Training loss curves
│   └── ...
└── analysis/                  # Belief state regression analysis
    ├── {sweep_id}_{run_id}/   # Analysis directory
    │   ├── checkpoint_*.joblib         # Regression results per checkpoint
    │   ├── ground_truth_data.joblib    # Neural network belief states
    │   ├── markov3_checkpoint_*.joblib # Classical Markov comparisons
    │   └── markov3_ground_truth_data.joblib # Classical belief states
    └── ...
```

### Model Mappings

| Model ID | Architecture | Process | Description |
|----------|--------------|---------|-------------|
| 20241121152808_55 | LSTM | Mess3 | LSTM trained on classical process |
| 20241205175736_23 | Transformer | Mess3 | Transformer trained on classical process |
| 20241121152808_63 | GRU | Mess3 | GRU trained on classical process |
| 20241121152808_71 | RNN | Mess3 | RNN trained on classical process |
| 20241121152808_53 | LSTM | FRDN | LSTM trained on quantum process |
| 20250422023003_1 | Transformer | FRDN | Transformer trained on quantum process |
| 20241121152808_61 | GRU | FRDN | GRU trained on quantum process |
| 20241121152808_69 | RNN | FRDN | RNN trained on quantum process |
| 20241121152808_49 | LSTM | Bloch Walk | LSTM trained on quantum process |
| 20241205175736_17 | Transformer | Bloch Walk | Transformer trained on quantum process |
| 20241121152808_57 | GRU | Bloch Walk | GRU trained on quantum process |
| 20241121152808_65 | RNN | Bloch Walk | RNN trained on quantum process |
| 20241121152808_48 | LSTM | Moon Process | LSTM trained on post-quantum process |
| 20250421221507_0 | Transformer | Moon Process | Transformer trained on post-quantum process |
| 20241121152808_56 | GRU | Moon Process | GRU trained on post-quantum process |
| 20241121152808_64 | RNN | Moon Process | RNN trained on post-quantum process |

## Process Descriptions

### Mess3 (Classical Process)
A classical stochastic process serving as a baseline for comparison with quantum processes.

### FRDN (Finite Random Dynamics Networks)
A quantum process representing finite random dynamics networks, modeling quantum systems with specific structural properties.

### Bloch Walk
A quantum random walk process on the Bloch sphere, representing quantum state evolution in a geometric framework.

### Moon Process
A post-quantum stochastic process that explores computational mechanics beyond standard quantum frameworks.

## Reproducibility Workflows

### Option 1: Public Reproduction (Recommended for External Users)

#### Prerequisites
```bash
# Install public dependencies
pip install -r requirements_public.txt

# Or install minimal requirements:
pip install torch numpy pandas scikit-learn matplotlib huggingface-hub datasets joblib tqdm
```

#### Generating Paper Figures

**Figure 2: Belief Grid Visualization**
```bash
# Use HuggingFace data (automatic download)
python Fig2.py --data-source huggingface

# Use local data (if available)
python Fig2.py --data-source local --data-dir path/to/analysis/files

# Auto-detect (try local first, fallback to HF)
python Fig2.py --data-source auto
```

**Figure 3: Multi-Panel Analysis**
```bash
python Fig3.py --data-source huggingface --output-dir my_figures/
```

**Figure 4: Representational Similarity Analysis**
```bash
python Fig4.py --data-source huggingface --checkpoint last --layer combined
```

#### Running Belief State Regression Analysis

For public users who want to reproduce the analysis from scratch:

```bash
# Run regression analysis using HuggingFace models
python scripts/activation_analysis/run_regression_analysis.py \
    --source huggingface \
    --repo-id SimplexAI/quantum-representations \
    --output-dir my_analysis_results \
    --only-final
```

### Option 2: Internal Reproduction (For Authors/Collaborators)

#### Prerequisites
```bash
# Install full dependencies
pip install -r pyproject.toml

# Set up AWS credentials for S3 access
export COMPANY_AWS_ACCESS_KEY_ID="your_key"
export COMPANY_AWS_SECRET_ACCESS_KEY="your_secret"
export COMPANY_AWS_DEFAULT_REGION="us-west-2"
export COMPANY_S3_BUCKET_NAME="your_bucket"
```

#### Running Analysis
```bash
# Use S3 data (internal)
python scripts/activation_analysis/run_regression_analysis.py --source s3

# Generate figures with local analysis results
python Fig2.py --data-source local --data-dir scripts/activation_analysis/run_predictions_RCOND_FINAL
```

## Data Access Examples

### Loading Analysis Data

```python
from scripts.data_manager import DataManager

# Initialize data manager with HuggingFace source
dm = DataManager(source='huggingface')

# Get analysis files for a specific model
model_files = dm.get_analysis_files('20241121152808_57')  # GRU Bloch Walk
print(f"Ground truth: {model_files['ground_truth']}")
print(f"Checkpoints: {model_files['checkpoints']}")

# Load regression results
import joblib
analysis_data = joblib.load(model_files['checkpoints'][0])
for layer, metrics in analysis_data.items():
    print(f"Layer {layer} RMSE: {metrics['rmse']}")
```

### Loading Model Checkpoints

```python
from scripts.huggingface_loader import HuggingFaceModelLoader

# Initialize HuggingFace model loader
loader = HuggingFaceModelLoader(repo_id='SimplexAI/quantum-representations')

# Load a specific model
sweep_id = "20241121152808"
run_name = "run_57_L4_H64_GRU_uni_tom_quantum"
checkpoints = loader.list_checkpoints(sweep_id, run_name)

# Load final checkpoint
model, config = loader.load_checkpoint(sweep_id, run_name, checkpoints[-1])
print(f"Model config: {config}")
```

### Programmatic Data Download

```python
from scripts.data_manager import DataManager

# Download entire dataset for offline use
dm = DataManager(source='huggingface')
dataset_path = dm.download_from_huggingface()
print(f"Dataset downloaded to: {dataset_path}")

# List all available models
models = dm.list_available_models()
for model_id in models:
    info = dm.get_model_info(model_id)
    print(f"{model_id}: {info['description']}")
```

## Command-Line Interface Reference

### Figure Generation Scripts

All figure scripts support the following common arguments:

- `--data-source {local,huggingface,auto}`: Data source selection
- `--data-dir PATH`: Local data directory (for local source)
- `--output-dir PATH`: Output directory for generated figures
- `--checkpoint {last,NUMBER}`: Target checkpoint for analysis
- `--layer STRING`: Target layer (e.g., 'combined', 'layer0')

### Analysis Scripts

`run_regression_analysis.py` supports:

- `--source {s3,huggingface}`: Model source
- `--repo-id REPO`: HuggingFace repository ID
- `--output-dir PATH`: Output directory for results
- `--device STRING`: Device for tensor operations
- `--splits INTEGER`: Number of k-fold splits
- `--only-final`: Process only initial and final checkpoints

## File Formats

### Model Checkpoints (.pt)
PyTorch model checkpoints containing:
- Model state dictionary
- Optimizer state (if available)
- Training metadata

### Analysis Files (.joblib)
Serialized analysis results containing:
- `checkpoint_*.joblib`: Layer-wise regression metrics (RMSE, MAE, R²)
- `ground_truth_data.joblib`: True belief states and probabilities
- `markov3_*.joblib`: Classical Markov model comparisons

### Configuration Files
- `run_config.yaml`: Training configuration parameters
- `loss.csv`: Training and validation loss curves

## Computational Requirements

### Minimum Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for dataset download
- **Python**: 3.8+ with scientific computing libraries

### For Training/Analysis
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (optional but recommended)
- **RAM**: 32GB recommended for full analysis
- **Storage**: 50GB+ for intermediate results

## Troubleshooting

### Common Issues

**HuggingFace Download Errors**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python Fig2.py --data-source huggingface
```

**Import Errors**
```bash
# Ensure scripts directory is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"
```

**Memory Issues**
```bash
# Use CPU instead of GPU for large models
python Fig2.py --data-source huggingface --device cpu
```

### Getting Help

For issues with reproduction:
1. Check the [repository issues](https://github.com/your-org/epsilon-transformers/issues)
2. Verify your environment matches `requirements_public.txt`
3. Ensure HuggingFace dataset access is working

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{epsilon-transformers-2024,
  title={Neural networks leverage nominally quantum and post-quantum representations},
  author={Your Authors},
  journal={Your Journal},
  year={2024},
  url={https://huggingface.co/datasets/SimplexAI/quantum-representations}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions about the methods or data access:
- Email: [your-email@domain.com]
- Repository: [https://github.com/your-org/epsilon-transformers]
- Dataset: [https://huggingface.co/datasets/SimplexAI/quantum-representations]