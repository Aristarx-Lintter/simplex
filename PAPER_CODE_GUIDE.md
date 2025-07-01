# Paper Code Guide

This document maps the methods described in the paper to their corresponding implementation files in the codebase.

## Section: Experimental Methods for Training Networks

### Stochastic Process Implementations
**Paper Reference**: Appendix - Example Processes (Mess3, FRDN, Bloch Walk, Moon Process)

**Code Files**:
- `/epsilon_transformers/process/transition_matrices.py`
  - `mess3()` - Mess3 process (classical)
  - `fanizza()` - FRDN process (quantum)
  - `tom_quantum()` - Bloch Walk process (quantum)
  - `post_quantum()` - Moon Process (post-quantum)
- `/epsilon_transformers/process/GHMM.py` - Base class for all processes
- `/epsilon_transformers/process/MixedStateTree.py` - Mixed state representation

### Model Architectures
**Paper Reference**: Section - Model Architectures

**Transformer Implementation**:
- Uses TransformerLens framework (external library)
- Configuration in training scripts: `/scripts/train.py`

**RNN/LSTM/GRU Implementation**:
- `/epsilon_transformers/training/networks.py` - Custom PyTorch implementations
- `/scripts/train_rnn.py` - RNN-specific training script

### Training Methodology
**Paper Reference**: Section - Training Methodology & Training Hyperparameters

**Training Scripts**:
- `/scripts/train.py` - Transformer training
- `/scripts/train_rnn.py` - RNN/LSTM/GRU training
- `/scripts/launcher_cuda_parallel.py` - Parallel execution framework

**Data Generation**:
- `/epsilon_transformers/training/generate_data.py` - Process data generation
- `/epsilon_transformers/training/dataloader.py` - Data loading and batching

**Configuration Files**:
- `/scripts/experiment_config_*.yaml` - Experiment configurations
- Example: `experiment_config_tom_quantum.yaml` contains process parameters

## Section: Analysis Methods to Probe for Belief Geometry

### General Approach
**Paper Reference**: Section - General Approach (weighted least squares regression)

**Core Implementation**:
- `/scripts/activation_analysis/regression.py`
  - `run_activation_to_beliefs_regression_kf()` - Main regression with K-fold CV
  - Implements weighted least squares with regularization

### Activation Extraction
**Paper Reference**: Implementation details - Extracting layer activations

**Code Files**:
- `/scripts/activation_analysis/data_loading.py` - `ActivationExtractor` class
- `/epsilon_transformers/analysis/activation_analysis.py` - Core analysis functions
- `/scripts/activation_analysis/config.py` - Layer definitions

### Belief State Computation
**Paper Reference**: Implementation details - Computing belief states

**Code Files**:
- `/scripts/activation_analysis/belief_states.py` - Belief state generation
- `/epsilon_transformers/analysis/activation_analysis.py` - `prepare_msp_data()`

### Cross-Validation and Model Selection
**Paper Reference**: Section - Cross-Validation and Model Selection

**Implementation**:
- K-fold CV in `/scripts/activation_analysis/regression.py`
- Regularization parameter selection via cross-validation
- Uses 10-fold CV as described in paper

### Evaluation Metrics
**Paper Reference**: Section - Evaluation Metrics (MSE, RMSE)

**Implementation**:
- Metrics calculated in regression functions
- RMSE computation throughout analysis scripts

### Cosine Similarity Analysis
**Paper Reference**: Section - Cosine Similarity Analysis

**Implementation**:
- Used in Figure 4 generation
- See `Fig4.py` - `calculate_cosine_similarities()`

### Control Experiments
**Paper Reference**: Section - Control Experiments

**Implementation**:
- Random baseline: Checkpoint 0 analysis
- Markov-3 approximations: `precompute_markov.py`

## Figure Generation Scripts

### Figure 2 (Main Paper)
**Script**: `Fig2.py` (copied from `visualize_belief_grid_clean.py`)
- Shows belief geometry visualization for 2 processes
- Includes RMSE bar charts

### Figure 3 (Main Paper)
**Script**: `Fig3.py`
- Multi-panel analysis figure
- Top row: Belief progression over training
- Bottom row: RMSE metrics, loss curves, layer analysis

### Figure 4 (Main Paper)
**Script**: `Fig4.py`
- Representational Similarity Analysis (RSA)
- 2x3 grid comparing Transformer and LSTM
- Shows cosine similarity between true and predicted beliefs

### Appendix Figures
**Script**: `FigAppendix.py` (copied from `visualize_belief_grid_clean_extended.py`)
- Extended belief geometry visualizations
- All 4 processes, all 4 model types

## Key Analysis Functions

### Main Analysis Runner
- `/scripts/activation_analysis/main.py` - Orchestrates full analysis pipeline

### Dimensionality Analysis
- `/scripts/activation_analysis/do_pca_analysis.py` - PCA analysis
- Cumulative variance calculation in analysis scripts

### Visualization Utilities
- `/epsilon_transformers/visualization/plots.py` - General plotting utilities
- `/scripts/activation_analysis/visualization.py` - Activation-specific plots

## Data and Checkpoints

### Process Data
- Pre-computed in `/process_data/` directory
- Contains transition matrices and probabilities

### Model Checkpoints
- Stored in sweep directories (e.g., `/data/20241121152808/`)
- Checkpoint files contain model weights and optimizer state

### Analysis Results
- Saved as `.joblib` files in `run_predictions_RCOND_FINAL/`
- Contains regression results, predictions, and metrics

## Reproducing Results

1. **Training Models**: Use experiment config files with training scripts
2. **Running Analysis**: Use `make_all_final_figures.py` or individual analysis scripts
3. **Generating Figures**: Run `Fig2.py`, `Fig3.py`, `Fig4.py`, `FigAppendix.py`

## Dependencies

- PyTorch 2.0
- TransformerLens (for transformer models)
- NumPy, SciPy, Scikit-learn
- Matplotlib for visualization
- Weights & Biases for experiment tracking