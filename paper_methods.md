# Epsilon-Transformers: Methods and Reproducibility Guide

This document describes the computational pipeline used to generate the activation analysis results presented in our paper "Neural networks leverage nominally quantum and post-quantum representations" and provides comprehensive methods for reproducing these results using both internal and public data sources.

## Dataset and Model Access

### Public Dataset
The complete dataset including trained models and analysis results is publicly available on HuggingFace:

**Repository**: [`SimplexAI/quantum-representations`](https://huggingface.co/datasets/SimplexAI/quantum-representations)

The dataset contains:
- **16 trained neural network models** (4 architectures × 4 processes)
- **Complete belief state regression analysis** with k-fold cross-validation results
- **Model checkpoints** at multiple training stages
- **Training configurations and loss curves**

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

## Overview

The activation analysis pipeline performs weighted least squares regression to map neural network activations to belief states (probability distributions over hidden states). The pipeline analyzes how well different neural architectures (Transformers, LSTMs, GRUs, RNNs) learn to represent beliefs about the underlying hidden Markov models (HMMs) generating the observed sequences.

## Pipeline Architecture

### Core Components

1. **Main Analysis Script** (`scripts/activation_analysis/run_regression_analysis.py`)
   - Standalone script for running the complete analysis
   - Processes predefined sweep/run pairs for multiple process types
   - Handles data deduplication, k-fold cross-validation, and result saving

2. **Data Loading** (`scripts/activation_analysis/data_loading.py`)
   - `ModelDataManager`: Handles model and checkpoint loading from S3
   - `ActivationExtractor`: Extracts activations from different model architectures
   - Implements caching for efficient data reuse

3. **Belief State Generation** (`scripts/activation_analysis/belief_states.py`)
   - `BeliefStateGenerator`: Creates theoretical belief states from Markov approximations
   - Generates belief states for Markov orders 1-4
   - Implements dimension limiting (max 64) to prevent computational explosion

4. **Regression Analysis** (`scripts/activation_analysis/regression.py`)
   - `RegressionAnalyzer`: Performs weighted least squares regression
   - Maps activations to belief states using various regularization parameters (rcond)
   - Implements efficient SVD-based pseudoinverse computation

5. **Configuration** (`scripts/activation_analysis/config.py`)
   - Centralized configuration for all pipeline parameters
   - Defines sweep IDs, device settings, and analysis parameters

6. **Utilities** (`scripts/activation_analysis/utils.py`)
   - Helper functions for standardization, variance analysis, and result saving
   - Logging configuration and data serialization

## Data Generation Process

### Step 1: Sweep and Run Selection
- The pipeline processes multiple sweeps for different process types:
  - **Mess3 Process**: 4 architectures (LSTM, Transformer, GRU, RNN)
  - **FRDN (Fanizza) Process**: 4 architectures
  - **Bloch Walk Process**: 4 architectures  
  - **Moon Process**: 4 architectures
- Multiple sweep IDs are used: `20241121152808` (RNN models), `20241205175736`, `20250422023003`, `20250421221507` (Transformer models)
- Only 4-layer networks (runs with 'L4' in the ID) are analyzed

### Step 2: Mixed State Presentation (MSP) Data Loading
For each run, the pipeline loads:
- **Input sequences**: Token sequences fed to the neural network
- **Neural belief states**: True posterior distributions given the input sequences
- **Word probabilities**: Weights for each data point based on generative model probabilities

### Step 3: Classical Belief State Generation
The pipeline generates Markov approximations:
- Focuses on Markov order 3 for the analysis (as seen in `run_regression_analysis.py`)
- Each order captures increasingly complex temporal dependencies
- Higher orders have exponentially more states (limited to 64 dimensions)
- Belief states are computed analytically from the Markov transition matrices

### Step 4: Activation Extraction
For each checkpoint:
- **Transformers**: Extracts activations from residual streams at each layer
- **RNNs**: Extracts hidden states and adds one-hot encoded inputs
- Activations are extracted for both neural and markov approximation belief inputs

### Step 5: Regression Analysis
The core analysis performs weighted least squares regression with k-fold cross-validation:

1. **Data Deduplication**:
   - Identifies and combines duplicate input prefixes
   - Sums probabilities for duplicate sequences
   - Reduces computational load while preserving statistical properties

2. **K-Fold Cross-Validation**:
   - Uses 10-fold cross-validation (N_SPLITS = 10)
   - Ensures robust performance estimates
   - Prevents overfitting to specific data splits

3. **Feature Analysis**:
   - Computes weighted PCA for variance explanation
   - Z-score normalization for standardized analysis
   - Tracks cumulative variance explained

4. **Weighted Regression**:
   - Weights samples by their probability under the generative model
   - Adds bias term to capture constant offsets
   - Uses efficient SVD-based pseudoinverse computation

5. **Regularization Sweep**:
   - Tests 53 different rcond values (regularization parameters)
   - Range: 1e-15 to 1e-3 in log scale
   - Identifies optimal regularization for each layer/target combination

6. **Performance Metrics**:
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - MSE (Mean Square Error)
   - R-squared (proportion of variance explained)
   - Distance metric (weighted L2 distance)

### Step 6: Checkpoint Processing
The analysis can process:
- **Initial and Final Only**: Default mode (ONLY_INITIAL_AND_FINAL = True)
  - Processes first checkpoint (epoch 0) and last checkpoint
- **All Checkpoints**: Optional mode for complete training trajectory analysis

### Step 7: Result Aggregation
For each run, the pipeline saves multiple files:
- `ground_truth_data.joblib`: Deduplicated neural network beliefs and probabilities
- `markov3_ground_truth_data.joblib`: Markov order 3 ground truth data
- `checkpoint_{ckpt_ind}.joblib`: Per-checkpoint results including:
  - Predicted beliefs
  - Performance metrics (RMSE, MAE, MSE, R², distance)
  - Variance explained (raw and z-scored)
  - Validation loss
- `markov3_checkpoint_{ckpt_ind}.joblib`: Classical model results for comparison

## Key Implementation Details

### Data Deduplication
- Identifies sequences with identical prefixes to avoid redundant computation
- Aggregates probabilities for duplicate sequences
- Maintains consistency checks for activation values

### Memory Management
- Processes data efficiently by deduplicating before analysis
- Saves predictions only for initial/final checkpoints by default
- Uses joblib for efficient serialization of large arrays

### Computational Optimizations
- K-fold cross-validation for robust performance estimates
- Efficient pseudoinverse computation with regularization sweep
- Combined layer activations for holistic analysis

## Configuration Parameters

Key settings used in the analysis:
- **N_SPLITS**: 10 (k-fold cross-validation splits)
- **DEVICE**: 'cuda' (configurable to 'cuda' or 'mps')
- **RANDOM_STATE**: 42 (for reproducibility)
- **ONLY_INITIAL_AND_FINAL**: True (process first and last checkpoints only)
- **RCOND_SWEEP_LIST**: 53 values from 1e-15 to 1e-3
- **Markov Order**: 3 (for classical belief comparison)
- **Activation layers analyzed**:
  - Transformers: Residual streams (pre/post) at each layer
  - RNNs: Hidden states at each layer plus input embeddings
  - Combined: Concatenation of all layer activations

## Output Structure

The output folder (now `belief_regression_results`) contains:
```
belief_regression_results/
├── {sweep_id}_{run_number}/           # e.g., 20241121152808_55
│   ├── ground_truth_data.joblib       # Deduplicated neural beliefs
│   ├── markov3_ground_truth_data.joblib # Markov order 3 beliefs
│   ├── checkpoint_0.joblib            # First checkpoint results
│   ├── checkpoint_{final}.joblib      # Final checkpoint results
│   ├── markov3_checkpoint_0.joblib    # Markov results for first checkpoint
│   └── markov3_checkpoint_{final}.joblib # Markov results for final 
```

## Markov Approximation Algorithm

The codebase implements a Markov approximation algorithm to create classical baseline models for comparison with neural network performance. This algorithm is central to understanding how well neural networks learn to represent belief states compared to interpretable classical models.

### Algorithm Overview

The Markov approximation algorithm takes a Generalized Hidden Markov Model (GHMM) and creates a finite-order Markov chain that approximates the original process. This approximation captures temporal dependencies up to a specified history length (the Markov order).

### Core Implementation

The main algorithm is implemented in `epsilon_transformers/process/GHMM.py:markov_approximation()` and works as follows:

1. **State Space Construction** (`epsilon_transformers/process/GHMM.py:234-250`):
   - Generates all possible token sequences of length equal to the Markov order
   - Filters sequences by their probability under the original GHMM
   - Only retains sequences with probability > `min_state_prob` (default: 1e-13)
   - Each retained sequence becomes a state in the Markov approximation

2. **Transition Matrix Computation** (`epsilon_transformers/process/GHMM.py:253-285`):
   - For each state (representing a history), computes transition probabilities to successor states
   - A transition from state `s1` to state `s2` exists if:
     - `s1[1:] == s2[:-1]` (sequences overlap appropriately)
     - The transition probability is computed as: `P(s2) / P(s1[:-1])`
   - Filters out transitions with probability < 1e-9 for numerical stability

### Mathematical Formulation

For a GHMM with transition tensor `T(a, h', h)` (probability of transitioning from hidden state `h` to `h'` while emitting symbol `a`), the Markov approximation of order `k` works as follows:

1. **State Definition**: Each state represents a sequence `[a₁, a₂, ..., aₖ]`

2. **Transition Probability**: For sequences `s = [a₁, ..., aₖ]` and `s' = [a₂, ..., aₖ, aₖ₊₁]`:
   ```
   P(s → s') = P(s') / P([a₁, ..., aₖ₋₁])
   ```
   where probabilities are computed using the original GHMM dynamics

3. **Belief States**: For a path through the Markov chain, the belief state is the posterior distribution over the original hidden states given the observed sequence