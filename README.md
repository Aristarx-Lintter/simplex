# Epsilon-Transformers: Neural Networks and Quantum Representations

This repository contains the code for reproducing the results in "Neural networks leverage nominally quantum and post-quantum representations".

### Installation

```bash
# Clone the repository (quantum-public branch)
git clone -b quantum-public https://github.com/adamimos/epsilon-transformers.git
cd epsilon-transformers

# Install UV package manager
pip install uv

# Install dependencies
uv sync
```


### 1. Generate Paper Figures

```bash
# Figure 2: Belief Grid Visualization (3 rows: Mess3, TomQA, Moon Process)
uv run python Fig2.py --data-source huggingface

# Figure 3: Multi-Panel Analysis
uv run python Fig3.py --data-source huggingface

# Figure 4: Representational Similarity Analysis
uv run python Fig4.py --data-source huggingface

# Appendix Figures: Extended Model Comparisons
uv run python FigAppendix.py --data-source huggingface --model-type all
```

All figures are saved to the `Figs/` directory.

### 2. Training Networks (Optional)

Model checkpoints are publically available on HuggingFace. If you wish to recreate the training of the models used in the paper run the commands below. Note that these scripts require a GPU, and were run on a H100 GPU. 

Before running the training scripts, you will need to set up your WandB account. You can do this by running the following command:
```bash
export WANDB_API_KEY=your_api_key
```

Then, you can run the training scripts:

```bash
# RNN experiments (LSTM, GRU, RNN across all processes)
uv run python ./scripts/launcher_cuda_parallel_rnn.py --config ./scripts/experiment_config_rnn.yaml

# Transformer experiments
uv run python ./scripts/launcher_cuda_parallel.py --config ./scripts/experiment_config_transformer_mess3_bloch.yaml
uv run python ./scripts/launcher_cuda_parallel.py --config ./scripts/experiment_config_transformer_moon.yaml
uv run python ./scripts/launcher_cuda_parallel.py --config ./scripts/experiment_config_transformer_frdn.yaml
```

Additionally, for completeness and reproducibility we include links to the exact github commits when all networks were trained, in the Appendix of the manuscript.

### 3. Run Regression Analysis Pipeline

The regression analysis pipeline was used to analyze the trained models and produce the regression results used in the paper. It is very computationally intensive, and was run on a H100 GPU. Because of the computational cost, we provide the results of running this pipeline as part of the HuggingFace dataset (see below). If you wish to look at the code, we ran this using the following command:

```bash
uv run python -m scripts.activation_analysis.run_regression_analysis
```

This code used a private AWS s3 bucket with model checkpoints. We provide the checkpoints on Huggingface. See below.

### Running Regression Analysis with Your Own S3 Bucket (Optional)

If you want to run the regression analysis pipeline yourself, you'll need to set up your own S3 bucket with the model checkpoints from HuggingFace.

**Step 1: Download checkpoints from HuggingFace**
The model checkpoints are available at [`SimplexAI/quantum-representations`](https://huggingface.co/datasets/SimplexAI/quantum-representations). You'll need to download and put them on S3 with the following structure.

**Step 2: Set up S3 bucket structure**
The S3 loader expects the following folder structure:
```
your-s3-bucket/
├── quantum_runs/                     # Company bucket prefix
│   ├── 20241121152808/              # Sweep directory
│   │   ├── run_55/                  # Individual run directory
│   │   │   ├── 0.pt                # Initial checkpoint
│   │   │   ├── 4075724800.pt       # Final checkpoint
│   │   │   └── [other checkpoints...]
│   │   ├── run_49/
│   │   └── [other runs...]
│   ├── 20241205175736/              # Another sweep
│   └── [other sweeps...]
```

**Step 3: Configure environment variables**
Create a `.env` file in your project root:
```env
COMPANY_AWS_ACCESS_KEY_ID=your_access_key_id
COMPANY_AWS_SECRET_ACCESS_KEY=your_secret_access_key
COMPANY_AWS_DEFAULT_REGION=us-east-1 # or your region
COMPANY_S3_BUCKET_NAME=your-bucket-name
```

**Step 4: Update S3 permissions**
Make sure your AWS credentials have the following S3 permissions:
- `s3:GetObject`
- `s3:ListBucket`
- `s3:GetBucketLocation`

**Step 5: Run the analysis**
```bash
uv run python -m scripts.activation_analysis.run_regression_analysis
```


## Data Sources

### HuggingFace Dataset: [`SimplexAI/quantum-representations`](https://huggingface.co/datasets/SimplexAI/quantum-representations)

The dataset contains complete trained models and analysis results:

```
SimplexAI/quantum-representations/
├── analysis/                              # Regression analysis results
│   ├── 20241121152808_55/                # LSTM trained on Mess3
│   │   ├── ground_truth_data.joblib      # Neural network belief states
│   │   ├── markov3_ground_truth_data.joblib # Classical Markov-3 beliefs
│   │   ├── checkpoint_0.joblib           # Initial checkpoint analysis
│   │   ├── checkpoint_4075724800.joblib  # Final checkpoint analysis
│   │   ├── markov3_checkpoint_0.joblib   # Classical analysis (initial)
│   │   └── markov3_checkpoint_*.joblib   # Classical analysis (other checkpoints)
│   ├── 20241205175736_23/                # Transformer trained on Mess3
│   ├── 20241121152808_49/                # LSTM trained on Bloch Walk
│   ├── 20241205175736_17/                # Transformer trained on Bloch Walk
│   ├── 20250421221507_0/                 # Transformer trained on Moon Process
│   ├── 20241121152808_48/                # LSTM trained on Moon Process
│   ├── 20250422023003_1/                 # Transformer trained on FRDN
│   └── [12 more model directories...]     # Complete 16-model suite
└── models/                               # Raw model checkpoints (.pt files)
    ├── 20241121152808_55/
    │   ├── 0.pt                         # Initial model weights
    │   ├── 4075724800.pt               # Final trained weights
    │   └── [intermediate checkpoints...]
    └── [other model directories...]
```

**Model Mapping**:
- **LSTM Models**: `20241121152808_*` (55=Mess3, 49=Bloch, 48=Moon, 53=FRDN)
- **Transformer Models**: `20241205175736_*` (23=Mess3, 17=Bloch), `20250421221507_0` (Moon), `20250422023003_1` (FRDN)
- **GRU Models**: `20241121152808_*` (63=Mess3, 57=Bloch, 56=Moon, 61=FRDN)
- **RNN Models**: `20241121152808_*` (71=Mess3, 65=Bloch, 64=Moon, 69=FRDN)

**Analysis Files**:
- `ground_truth_data.joblib`: Neural network belief states and probabilities
- `markov3_ground_truth_data.joblib`: Classical Markov-3 approximation beliefs
- `checkpoint_*.joblib`: Regression results mapping activations to beliefs
- Each file contains RMSE, predicted beliefs, and performance metrics

### Local Training
Train your own models using the config files above. Results will be saved to `./results/` and can be analyzed with the regression pipeline.

## Repository Structure

```
├── Fig2.py, Fig3.py, Fig4.py, FigAppendix.py  # Paper figure generation
├── scripts/
│   ├── experiment_config_*.yaml              # Training configurations
│   ├── launcher_cuda_parallel.py             # GPU training launcher
│   └── activation_analysis/                  # Regression analysis pipeline
├── epsilon_transformers/                     # Core library code
└── Figs/                                     # Generated figures output
```

## More Details

For detailed methods and implementation details, see [`paper_methods.md`](paper_methods.md).

## Citation

If you use this code, please cite:

```bibtex
@article{epsilon_transformers,
  title={Neural networks leverage nominally quantum and post-quantum representations},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```