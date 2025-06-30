# Methods: Belief State Regression from Neural Network Activations

## Overview

We present a method for mapping neural network activations to belief states using weighted ridge regression with k-fold cross-validation. Our approach quantifies how well transformer and RNN architectures encode probabilistic beliefs about hidden Markov processes during sequence modeling.

## Problem Formulation

Given a trained neural network processing sequences from a hidden Markov model (HMM), we aim to recover the optimal Bayesian belief states from the network's internal activations. Let:

- $\mathbf{A}_t \in \mathbb{R}^D$ denote the activation vector at position $t$
- $\mathbf{b}_t \in \mathbb{R}^S$ denote the true belief state (posterior distribution over hidden states)
- $p_t$ denote the probability of observing the prefix up to position $t$

Our goal is to learn a linear mapping $f: \mathbb{R}^D \rightarrow \mathbb{R}^S$ such that $f(\mathbf{A}_t) \approx \mathbf{b}_t$.

## Data Preprocessing

### Prefix Deduplication

To handle the exponential growth of possible prefixes and ensure proper weighting, we deduplicate sequences based on unique prefixes. For each unique prefix $\pi$:

1. Identify all occurrences $(i, t)$ where sequence $i$ at position $t$ has prefix $\pi$
2. Verify activation consistency: $\|\mathbf{A}_{i,t} - \mathbf{A}_{j,t'}\| < \epsilon$ for all occurrences
3. Aggregate probabilities: $p_\pi = \sum_{(i,t) \in \pi} p_{i,t}$

This deduplication ensures that frequently occurring prefixes are properly weighted in the regression without redundant computation.

## Weighted Ridge Regression

We formulate the belief state recovery as a weighted least squares problem with $L_2$ regularization:

$$\min_{\mathbf{W}, \mathbf{b}} \sum_{\pi} p_\pi \|\mathbf{W}\mathbf{A}_\pi + \mathbf{b} - \mathbf{b}_\pi\|^2 + \lambda \|\mathbf{W}\|_F^2$$

where:
- $\mathbf{W} \in \mathbb{R}^{S \times D}$ is the weight matrix
- $\mathbf{b} \in \mathbb{R}^S$ is the bias vector
- $p_\pi$ is the normalized probability weight for prefix $\pi$
- $\lambda$ is the regularization parameter

### Efficient Implementation

We reformulate the problem using weighted design matrices. Let $\mathbf{X} = [\mathbf{1}, \mathbf{A}]$ be the augmented activation matrix and $\mathbf{P} = \text{diag}(\sqrt{p_1}, ..., \sqrt{p_N})$ be the square root of the probability weights. The weighted normal equations become:

$$(\mathbf{X}^T\mathbf{P}^2\mathbf{X} + \lambda\mathbf{I})[\mathbf{b}; \mathbf{W}^T] = \mathbf{X}^T\mathbf{P}^2\mathbf{Y}$$

We solve this efficiently using SVD-based pseudoinverse computation:

1. Compute $\mathbf{A} = \mathbf{X}^T\mathbf{P}^2\mathbf{X}$
2. Perform eigendecomposition: $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{U}^T$
3. Compute regularized pseudoinverse: $\mathbf{A}^+ = \mathbf{U}\mathbf{\Sigma}^+\mathbf{U}^T$

where $\mathbf{\Sigma}^+$ is computed with truncation parameter `rcond`: $\sigma_i^+ = 1/\sigma_i$ if $\sigma_i > \text{rcond} \cdot \sigma_{\max}$, else $0$.

## K-Fold Cross-Validation

We employ 10-fold cross-validation to select the optimal regularization parameter:

1. **Data Splitting**: Partition deduplicated prefixes into $k=10$ folds, maintaining probability weights
2. **Hyperparameter Sweep**: For each `rcond` value in $\{10^{-15}, 10^{-14}, ..., 10^{-1}\}$:
   - For each fold $i$:
     - Train on folds $\{1, ..., k\} \setminus \{i\}$
     - Evaluate weighted error on fold $i$: $E_i = \sum_{j \in \text{fold}_i} p_j \|\hat{\mathbf{b}}_j - \mathbf{b}_j\|$
   - Compute average error: $\bar{E}_{\text{rcond}} = \frac{1}{k}\sum_{i=1}^k E_i$
3. **Selection**: Choose `rcond*` = $\arg\min_{\text{rcond}} \bar{E}_{\text{rcond}}$

## Final Model Training

After hyperparameter selection, we train the final model on the entire dataset using the optimal `rcond*`:

1. Compute the full weighted design matrix
2. Solve the regularized normal equations with `rcond*`
3. Evaluate performance metrics on the full dataset

## Evaluation Metrics

We assess regression quality using multiple metrics, all weighted by prefix probabilities:

- **Weighted RMSE**: $\text{RMSE} = \sqrt{\sum_\pi p_\pi \|\hat{\mathbf{b}}_\pi - \mathbf{b}_\pi\|^2}$
- **Weighted MAE**: $\text{MAE} = \sum_\pi p_\pi \|\hat{\mathbf{b}}_\pi - \mathbf{b}_\pi\|_1 / S$
- **Weighted R²**: $R^2 = 1 - \frac{\sum_\pi p_\pi \|\hat{\mathbf{b}}_\pi - \mathbf{b}_\pi\|^2}{\sum_\pi p_\pi \|\mathbf{b}_\pi - \bar{\mathbf{b}}\|^2}$

where $\bar{\mathbf{b}} = \sum_\pi p_\pi \mathbf{b}_\pi$ is the weighted mean belief state.

## Multi-Layer Analysis

We perform regression analysis at multiple levels:

1. **Individual Layers**: Each transformer/RNN layer's activations separately
2. **Combined Representation**: Concatenated activations from all layers
3. **Combined without Input**: Concatenated activations excluding embedding layers

This allows us to assess how belief state information is distributed across the network architecture.

## Computational Considerations

- **Device Acceleration**: Computations utilize GPU/MPS acceleration when available
- **Numerical Stability**: SVD-based pseudoinverse ensures stability for ill-conditioned matrices
- **Memory Efficiency**: Deduplication reduces memory requirements by orders of magnitude
- **Weighted PCA**: We additionally compute weighted principal component analysis to assess the intrinsic dimensionality of activation spaces

This methodology enables systematic comparison of how different neural architectures encode probabilistic beliefs during sequence modeling tasks.