# Methods: Belief State Regression Analysis

## Overview

We present a method for predicting belief states from neural network activations using weighted ridge regression with k-fold cross-validation. Our approach addresses the challenge of learning a mapping from high-dimensional activation spaces to probability distributions over hidden states, while accounting for the non-uniform sampling of trajectories in the training data.

## Problem Formulation

Given a dataset of neural network activations and corresponding belief states:
- **Activations**: $\mathbf{X} \in \mathbb{R}^{N \times D}$, where $N$ is the number of samples and $D$ is the activation dimension
- **Belief states**: $\mathbf{Y} \in \mathbb{R}^{N \times S}$, where $S$ is the number of hidden states
- **Sample weights**: $\mathbf{w} \in \mathbb{R}^{N}$, representing the probability of each trajectory

Our goal is to learn a linear mapping $\mathbf{\beta} \in \mathbb{R}^{(D+1) \times S}$ that minimizes the weighted prediction error.

## Data Preprocessing

### Trajectory Deduplication

To handle repeated trajectories in the dataset, we implement a deduplication strategy based on sequence prefixes:

1. **Prefix Extraction**: For each trajectory of length $T$, we extract all prefixes of length $t \in \{1, 2, ..., T\}$
2. **Unique Prefix Identification**: We identify unique prefixes across all trajectories
3. **Weight Aggregation**: For duplicate prefixes, we sum their associated weights: $w_{\text{unique}} = \sum_{i \in \text{duplicates}} w_i$

This ensures that common trajectory prefixes are not overrepresented in the training data.

## Weighted Ridge Regression

We formulate the regression problem as a weighted least squares optimization with $L_2$ regularization:

$$\mathbf{\beta}^* = \arg\min_{\mathbf{\beta}} \sum_{i=1}^{N} w_i \|\mathbf{y}_i - \mathbf{\tilde{x}}_i^T \mathbf{\beta}\|_2^2 + \lambda \|\mathbf{\beta}\|_2^2$$

where $\mathbf{\tilde{x}}_i = [1, \mathbf{x}_i^T]^T$ includes a bias term, and $\lambda$ is controlled by the regularization parameter `rcond`.

### Closed-form Solution

The weighted ridge regression has a closed-form solution. First, we transform the problem using weight square roots:

$$\mathbf{X}_w = \text{diag}(\sqrt{\mathbf{w}}) \mathbf{\tilde{X}}, \quad \mathbf{Y}_w = \text{diag}(\sqrt{\mathbf{w}}) \mathbf{Y}$$

The solution is then:

$$\mathbf{\beta}^* = (\mathbf{X}_w^T \mathbf{X}_w + \lambda \mathbf{I})^{-1} \mathbf{X}_w^T \mathbf{Y}_w$$

### Efficient Computation via SVD

To ensure numerical stability and computational efficiency, we compute the pseudoinverse using Singular Value Decomposition (SVD):

1. Compute SVD: $\mathbf{A} = \mathbf{X}_w^T \mathbf{X}_w = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
2. For each regularization parameter $\text{rcond} \in \mathcal{R}$:
   - Threshold singular values: $\sigma_i' = \begin{cases} \sigma_i & \text{if } \sigma_i \geq \text{rcond} \cdot \sigma_{\max} \\ 0 & \text{otherwise} \end{cases}$
   - Compute pseudoinverse: $\mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T$

This approach allows efficient evaluation of multiple regularization parameters without repeated matrix decompositions.

## K-Fold Cross-Validation

We employ k-fold cross-validation to select the optimal regularization parameter:

### Fold Generation
1. Sort trajectories by their final position in the sequence
2. Apply stratified sampling to ensure each fold contains a representative distribution of sequence positions
3. Split data into $K$ folds (default $K=5$)

### Hyperparameter Selection
For each fold $k \in \{1, ..., K\}$ and each $\text{rcond} \in \mathcal{R}$:

1. **Train**: Fit regression model on folds $\{1, ..., K\} \setminus \{k\}$
2. **Evaluate**: Compute weighted test error on fold $k$:
   $$E_k(\text{rcond}) = \sum_{i \in \text{fold}_k} w_i \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2$$

3. **Average**: Compute mean error across folds:
   $$\bar{E}(\text{rcond}) = \frac{1}{K} \sum_{k=1}^{K} E_k(\text{rcond})$$

4. **Select**: Choose optimal parameter:
   $$\text{rcond}^* = \arg\min_{\text{rcond} \in \mathcal{R}} \bar{E}(\text{rcond})$$

### Final Model Training

After selecting $\text{rcond}^*$, we train the final model on the entire dataset using the optimal regularization parameter.

## Evaluation Metrics

We evaluate model performance using multiple metrics:

### 1. Weighted Root Mean Square Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{\sum_{i=1}^{N} w_i \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2^2}{\sum_{i=1}^{N} w_i}}$$

### 2. Weighted Mean Absolute Error (MAE)
$$\text{MAE} = \frac{\sum_{i=1}^{N} w_i \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_1}{\sum_{i=1}^{N} w_i}$$

### 3. Weighted R² Score
$$R^2 = 1 - \frac{\sum_{i=1}^{N} w_i \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2^2}{\sum_{i=1}^{N} w_i \|\mathbf{y}_i - \bar{\mathbf{y}}_w\|_2^2}$$

where $\bar{\mathbf{y}}_w = \frac{\sum_{i=1}^{N} w_i \mathbf{y}_i}{\sum_{i=1}^{N} w_i}$ is the weighted mean.

### 4. Normalized Distance
For compatibility with previous analyses, we also report:
$$\text{norm\_dist} = \sum_{i=1}^{N} w_i \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2$$

## Implementation Details

### Regularization Parameter Grid
We evaluate $\text{rcond} \in \{10^{-15}, 10^{-14}, ..., 10^{-2}, 10^{-1}\}$ on a logarithmic scale, with additional option for sklearn's LinearRegression as a fallback.

### Computational Optimizations
1. **Batch SVD Computation**: Precompute SVD decomposition once and generate all pseudoinverses efficiently
2. **Parallel Processing**: Process multiple layers and targets concurrently
3. **Memory Management**: Use sparse representations where applicable

### Handling Edge Cases
- **Numerical Instability**: Fallback to sklearn's LinearRegression when SVD-based methods fail
- **Degenerate Solutions**: Skip folds that produce infinite or NaN errors
- **Small Sample Sizes**: Adjust number of folds when $N < 5K$

## Algorithm Summary

```
Algorithm: Weighted Ridge Regression with K-Fold CV for Belief State Prediction

Input: Activations X, Beliefs Y, Weights w, K folds, rcond values R
Output: Trained model β*, evaluation metrics

1. Preprocess data:
   - Deduplicate trajectories based on prefixes
   - Normalize weights: w ← w / sum(w)

2. Generate K stratified folds

3. Cross-validation:
   For each rcond in R:
     For each fold k:
       - Train on K-1 folds using weighted ridge regression
       - Evaluate on fold k
     - Compute average error across folds

4. Select optimal rcond* = argmin(average errors)

5. Train final model on full dataset with rcond*

6. Compute evaluation metrics (RMSE, MAE, R², norm_dist)

7. Return β*, metrics
```

This methodology enables robust prediction of belief states from neural network activations while accounting for the statistical properties of the training data and ensuring numerical stability through careful regularization and cross-validation.