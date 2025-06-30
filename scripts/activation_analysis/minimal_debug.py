"""
Minimal example for debugging the activation analysis pipeline.
This script uses VS Code cells to allow interactive execution and debugging.
Each section is marked with # %% and can be run independently in VS Code.
"""
# %% Imports
import os
# Note: If you see a linter error for torch, ensure it's installed: pip install torch
import torch  # This may need to be installed if not already available
import numpy as np
import pandas as pd
import plotly.graph_objs as go  # For visualization
from plotly.subplots import make_subplots  # For creating subplot figures
from collections import defaultdict  # For grouping data
import time

# Import directly from epsilon_transformers
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from epsilon_transformers.visualization.plots import _project_to_simplex  # For simplex projection
from epsilon_transformers.training.networks import create_RNN


# Import BeliefStateGenerator
from scripts.activation_analysis.belief_states import BeliefStateGenerator

# %% 1. Explore available sweeps, runs, and checkpoints
print("Exploring available sweeps, runs, and checkpoints...")
s3_loader = S3ModelLoader(use_company_credentials=True)

# List all available sweeps
print("\nAvailable sweeps:")
sweeps = s3_loader.list_sweeps()
for i, sweep in enumerate(sweeps):
    print(f"{i+1}. {sweep}")

# Focus on the specific sweep we're interested in
target_sweep = "20241205175736"
print(f"\nExploring runs for sweep {target_sweep}:")
runs = s3_loader.list_runs_in_sweep(target_sweep)
for i, run in enumerate(runs):
    print(f"{i+1}. {run}")

# Select the first run
first_run = runs[8]  # This should be "run_0_L4_H8_DH8_DM64_post_quantum" based on the output
print(f"\nSelected run: {first_run}")

# Get checkpoints for this run
print(f"Available checkpoints for {target_sweep}/{first_run}:")
checkpoints = s3_loader.list_checkpoints(target_sweep, first_run)
if len(checkpoints) > 20:
    print(f"Found {len(checkpoints)} checkpoints. Showing first 10:")
    for i, checkpoint in enumerate(checkpoints[:10]):
        print(f"{i+1}. {checkpoint}")
    print("...")
    last_idx = len(checkpoints) - 1
    print(f"{last_idx + 1}. {checkpoints[last_idx]}")
else:
    for i, checkpoint in enumerate(checkpoints):
        print(f"{i+1}. {checkpoint}")

# Take the last checkpoint
latest_checkpoint = checkpoints[-1]
print(f"\nSelected last checkpoint: {latest_checkpoint}")

# Extract step number
ckpt_filename = os.path.basename(latest_checkpoint)
step = int(ckpt_filename.replace(".pt", ""))
print(f"Step number: {step}")

# %% 2. Configure analysis parameters
# Use the exact values we discovered in the previous section
sweep_id = target_sweep
run_id = first_run
checkpoint = latest_checkpoint
model_type = "transformer"  # Hardcoded since we know it's a transformer
device = "cpu"  # Hardcoded to CPU for debugging

print(f"Analysis parameters:")
print(f"  Sweep: {sweep_id}")
print(f"  Run: {run_id}")
print(f"  Checkpoint: {checkpoint}")
print(f"  Model type: {model_type}")
print(f"  Device: {device}")

# %% 3. Load Model at Checkpoint
print("\nLoading model at checkpoint...")

# Directly load the model using the s3_loader
model, run_config = s3_loader.load_checkpoint(sweep_id, run_id, checkpoint, device=device)

# Verify the model
print("Model loaded successfully.")
print(f"Model configuration: {run_config}")
print(f"Model type: {type(model).__name__}")


# %%

# %% 4. Generate Belief States
print("\nGenerating Belief States...")

# First, prepare neural network beliefs using prepare_msp_data
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data

print("Getting neural network beliefs...")

# Get context length from run_config
n_ctx = run_config['model_config']['n_ctx']
run_config['n_ctx'] = n_ctx
nn_inputs, nn_beliefs, nn_indices, nn_probs, nn_unnormalized = prepare_msp_data(run_config, run_config['process_config'])
print(f"Neural network beliefs shape: {nn_beliefs.shape}")
print(f"Sample neural network beliefs: {nn_beliefs[0, :5]}")  # First 5 elements


# %%
from scripts.activation_analysis.data_loading import ActivationExtractor
# %%
act_extractor = ActivationExtractor(device)

# %%
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
nn_acts = act_extractor.extract_activations(model, nn_inputs, model_type, TRANSFORMER_ACTIVATION_KEYS)

# Print the shape of each activation
for key in nn_acts.keys():
    print(f"{key}: {nn_acts[key].shape}")

# Concatenate all activations into a single tensor
all_acts = []
for key in nn_acts.keys():
    # Reshape to (batch_size, sequence_length, -1)
    act_shape = nn_acts[key].shape
    reshaped_act = nn_acts[key].reshape(act_shape[0], act_shape[1], -1)
    all_acts.append(reshaped_act)

# Concatenate along the last dimension
concat_acts = torch.cat(all_acts, dim=2)
print(f"Concatenated activations shape: {concat_acts.shape}")

# %%
from scripts.activation_analysis.regression import RegressionAnalyzer, run_single_rcond_sweep_with_predictions

# %%
regression_analyzer = RegressionAnalyzer(device=device, use_efficient_pinv=True)

# %%
from scripts.activation_analysis.config import RCOND_SWEEP_LIST
nn_probs_uniform = torch.ones_like(nn_probs) / nn_probs.sum()
best_results = run_single_rcond_sweep_with_predictions(regression_analyzer, 
                                                       #concat_acts,
                                                       nn_acts['ln_final.hook_normalized'],
                                                       nn_beliefs, nn_probs/nn_probs.sum(), rcond_values=RCOND_SWEEP_LIST)

# %%
print(best_results.keys())
# %%
print(best_results['predictions'].shape)
# %%
# plot the predictions
import plotly.express as px
import numpy as np
results = best_results['predictions'].reshape(-1, 3)
# Check if best_results is None before trying to access it
if best_results is not None:
    # Create a DataFrame for plotly
    fig = px.scatter_3d(
        x=results[:, 0],
        y=results[:, 1],
        z=results[:, 2],
        #color=[0,0,0],#results[:, 2],
        opacity=.1,
        size_max=1,  # Set maximum marker size
        size=[1] * len(results)  # Set all points to small size
    )
    
    fig.update_layout(
        title="3D Visualization of Predictions",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        # Make markers smaller
        scene_camera_eye=dict(x=1.5, y=1.5, z=1.5)
    )
    
    fig.update_traces(marker=dict(size=.5))  # Set marker size to be very small
    
    fig.show()
else:
    print("Error: best_results is None. Cannot create plot.")
# %%
# make the same plot with nn_beliefs
nn_beliefs_reshaped = nn_beliefs.reshape(-1, 3)

# Reshape nn_probs to match the points in the scatter plot
nn_probs_reshaped = nn_probs.reshape(-1)
nn_probs_reshaped = nn_probs_reshaped/nn_probs_reshaped.sum()

# Convert tensors to numpy arrays for processing
nn_beliefs_np = nn_beliefs_reshaped.detach().cpu().numpy()
nn_probs_np = nn_probs_reshaped.detach().cpu().numpy()

# Use pandas to group by coordinates and aggregate probabilities
import pandas as pd
df = pd.DataFrame({
    'x': nn_beliefs_np[:, 0],
    'y': nn_beliefs_np[:, 1],
    'z': nn_beliefs_np[:, 2],
    'weight': nn_probs_np  # Using the same probability weights
})

# -----------------------------------------------------------------------------
# Compute Normalized RGB Components from Belief Coordinates
# -----------------------------------------------------------------------------
# Extract coordinates from your beliefs array
x = nn_beliefs_np[:, 0]   # used for R
y = nn_beliefs_np[:, 1]   # used for G (and as x-axis for plotting)
z = nn_beliefs_np[:, 2]   # used for plotting as y-axis (for the Y-Z projection)

# Compute B as the distance in the xy-plane
B_val = np.sqrt(x**2 + y**2)

# Compute global min/max for each channel
epsilon = 1e-8
min_R, max_R = x.min(), x.max()
min_G, max_G = y.min(), y.max()
min_B, max_B = B_val.min(), B_val.max()

# Normalize each channel to [0,1]
R_norm = (x - min_R) / (max_R - min_R + epsilon)
G_norm = (y - min_G) / (max_G - min_G + epsilon)
B_norm = (B_val - min_B) / (max_B - min_B + epsilon)

# Convert normalized values to 0-255 integers
R_int = (R_norm * 255).astype(int)
G_int = (G_norm * 255).astype(int)
B_int = (B_norm * 255).astype(int)

# -----------------------------------------------------------------------------
# Compute Floating Point Alpha Based on Probability
# -----------------------------------------------------------------------------
def compute_alpha_float(prob, scale=10, min_alpha=0.2):
    """
    Compute an alpha value (float in [0,1]) from a probability value (0-1).
    Uses a logarithmic transformation so that even very low probabilities are
    boosted to at least min_alpha.
    """
    transformed = np.log1p(prob * scale) / np.log1p(scale)
    alpha = min_alpha + (1 - min_alpha) * transformed
    return np.clip(alpha, 0, 1)

alpha_float = compute_alpha_float(nn_probs_np, scale=1, min_alpha=0.15)

# -----------------------------------------------------------------------------
# Create RGBA Strings with Floating Point Alpha
# -----------------------------------------------------------------------------
colors = [f"rgba({r},{g},{b},{alpha:.2f})" 
          for r, g, b, alpha in zip(R_int, G_int, B_int, alpha_float)]

# -----------------------------------------------------------------------------
# Create side-by-side Plotly Scattergl Plots with clean white background
# -----------------------------------------------------------------------------
# Create subplot figure with two side-by-side plots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Belief States", "Predictions"))

# Beliefs plot (left subplot)
x_beliefs, y_beliefs = _project_to_simplex(nn_beliefs.reshape(-1, 3))
beliefs_trace = go.Scattergl(
    x=x_beliefs,
    y=y_beliefs,
    mode='markers',
    marker=dict(
        color=colors,
        size=2
    ),
    name="Belief States"
)
fig.add_trace(beliefs_trace, row=1, col=1)

# Predictions plot (right subplot)
x = pred_df['x'].values
y = pred_df['y'].values
z = pred_df['z'].values

# Compute alpha values for predictions
pred_probs_np = pred_df['weight'].values
alpha_float_pred = compute_alpha_float(pred_probs_np, scale=1, min_alpha=0.15)

# Create RGBA colors for predictions
pred_colors = [f"rgba({r},{g},{b},{alpha:.2f})" 
               for r, g, b, alpha in zip(R_int, G_int, B_int, alpha_float_pred)]

x_preds, y_preds = _project_to_simplex(pred_df[['x', 'y', 'z']].values)

# Create the scatter plot for predictions
pred_trace = go.Scattergl(
    x=x_preds,
    y=y_preds,
    mode='markers',
    marker=dict(
        color=pred_colors,
        size=1
    ),
    name="Predictions"
)
fig.add_trace(pred_trace, row=1, col=2)

# Create clean white background with no grid, labels or numbers
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
    margin=dict(l=0, r=0, t=30, b=0),
    height=200,
    width=400
)

# Update axes for both subplots
for i in [1, 2]:
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="",
        row=1, col=i
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="",
        row=1, col=i
    )

# Show the plot
fig.show()