# Experimental Methods

## Experimental Design

We conduct a comprehensive evaluation of four neural network architectures on four distinct stochastic processes, resulting in 16 experimental configurations. The architectures include transformers, LSTMs, GRUs, and vanilla RNNs, while the processes consist of Mess3 (classical), FRDN (quantum), Bloch Walk (quantum), and the Moon Process (post-quantum)—each representing different computational complexity classes as described in Section [X].

## Model Architectures

For the Transformer architecture, we employ a 4-layer model implemented using the TransformerLens framework (Nanda & Bloom, 2022). The model uses multi-head attention with 4 heads (dimension 16 per head), 64-dimensional embeddings, and a 256-dimensional feed-forward network with ReLU activation. Layer normalization is applied before each sub-layer, and the model processes fixed sequences of 8 tokens with learned positional embeddings.

We compare three RNN variants, each configured with identical hyperparameters to ensure fair comparison: 4 recurrent layers with 64 hidden units per layer, unidirectional processing, one-hot input encoding, and a linear output projection. The variants differ only in their gating mechanisms: LSTM uses forget, input, and output gates; GRU employs reset and update gates; and the vanilla RNN uses simple tanh activation without gating.

## Training Methodology

Training data is generated from each stochastic process with the following parameters:
- **Mess3**: a=0.85, x=0.05
- **Fanizza**: α=2000.0, λ=0.49  
- **Tom Quantum**: α=1.0, β=7.14
- **Post Quantum**: α=2.72, β=0.5

Each training example consists of an 8-token input sequence with corresponding next-token targets for each position. All experiments use consistent random seeding (seed=42) for reproducibility.

### Training Hyperparameters

The table below shows the core hyperparameters used across all experiments. All models are trained for 20,000 epochs using the Adam optimizer with learning rate 1×10⁻⁴.

| **Hyperparameter** | **Value** |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 1×10⁻⁴ |
| Weight decay | None |
| Gradient clipping | None |
| Epochs | 20,000 |
| Validation frequency | Every epoch |
| Checkpoint frequency | Every 100 epochs |
| Random seed | 42 |

The majority of experiments use the following configuration:

| **Configuration** | **Standard Setting** |
|---|---|
| Batch size | 128 |
| Batches per epoch | 200 |
| LR scheduler | ReduceLROnPlateau* |
| Total checkpoints | 201 |

*ReduceLROnPlateau parameters: factor=0.5, patience=1000, cooldown=200, threshold=1×10⁻⁶

### Experiment-Specific Variations

While all RNN variants (LSTM, GRU, RNN) use the standard configuration above for all processes, we found that certain transformer experiments trained better with modified settings:

| **Experiment** | **Batch Size** | **Batches/Epoch** | **LR Scheduler** |
|---|---|---|---|
| Transformer-Fanizza | 16 | 20 | None |
| Transformer-Post Quantum | 16 | 20 | ReduceLROnPlateau |

We train using standard cross-entropy loss. Validation is performed every epoch on the full dataset, with model checkpoints saved every 100 epochs (201 total) and comprehensive metric logging via Weights & Biases.

## Implementation Details

All experiments are implemented in PyTorch 2.0 with CUDA acceleration, using FP32 precision throughout. Training is distributed across multiple GPUs, with specific GPU assignments managed through a parallel execution framework. To ensure reproducibility, we use fixed random seeds, deterministic data generation, consistent initialization schemes, and version-controlled configuration files.

Several key methodological decisions guide our experimental design. We maintain 4 layers across all architectures to ensure fair comparison of inductive biases rather than capacity differences. The high checkpoint frequency (every 100 epochs) enables detailed analysis of learning dynamics and convergence behavior. Finally, we deliberately avoid dropout, weight decay, or other regularization techniques to study the pure learning dynamics of each architecture on these processes.