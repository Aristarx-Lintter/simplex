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

# Import directly from epsilon_transformers
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
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
first_run = runs[6]  # This should be "run_0_L4_H8_DH8_DM64_post_quantum" 

# based on the output of the previous cell
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
    'prob': nn_probs_np
})

# Group by coordinates and sum probabilities
# Round coordinates to handle tiny numerical differences that might prevent exact matches
df_rounded = df.round(decimals=10)  # Adjust decimals as needed for your data
aggregated_df = df_rounded.groupby(['x', 'y', 'z']).agg(
    total_prob=('prob', 'sum'),
    count=('prob', 'count')
).reset_index()

print(f"Original points: {len(df)}, Unique locations: {len(aggregated_df)}")
print(f"Points with overlaps: {len(aggregated_df[aggregated_df['count'] > 1])}")

# Filter to keep only the top 50th percentile of aggregated probability
percentile_threshold = 99  # Show top 50%
prob_threshold = np.percentile(aggregated_df['total_prob'], 100 - percentile_threshold)
filtered_df = aggregated_df[aggregated_df['total_prob'] >= prob_threshold]

print(f"Showing {len(filtered_df)} points (top {percentile_threshold}% by probability)")
print(f"Probability threshold: {prob_threshold:.6f}")

# Create scatter plot with filtered data
fig = px.scatter_3d(
    filtered_df,
    x='x',
    y='y',
    z='z',
    color='total_prob',  # Color by total probability
    color_continuous_scale='Viridis',  # Choose a colorscale
    hover_data=['total_prob', 'count']  # Show these values on hover
)

# Set a fixed opacity and size for all points
fig.update_traces(marker=dict(opacity=0.003, size=4))

fig.update_layout(
    title=f"Top {percentile_threshold}% of Belief States by Probability",
    scene=dict(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        zaxis_title="Dimension 3"
    ),
    scene_camera_eye=dict(x=1.5, y=1.5, z=1.5),
    coloraxis_colorbar=dict(title="Total Probability")
)

fig.show()
# %%
print(nn_probs.shape)
print(nn_beliefs.shape)

# %%
# Alternative visualization using datashader for better aggregation
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
from datashader.mpl_ext import dsshow
from datashader.colors import Hot, inferno
import matplotlib.pyplot as plt
from functools import partial

# Convert to pandas DataFrame for datashader
nn_beliefs_np = nn_beliefs_reshaped.detach().cpu().numpy()
nn_probs_np = nn_probs_reshaped.detach().cpu().numpy()

ds_df = pd.DataFrame({
    'x': nn_beliefs_np[:, 0],
    'y': nn_beliefs_np[:, 1],
    'z': nn_beliefs_np[:, 2],
    'prob': nn_probs_np
})

# Create 2D projections with datashader (can be extended to 3D with additional packages)
print("Creating datashader visualization...")

# Set up the figure with three 2D projections
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# X-Y projection (top-down view)
canvas1 = ds.Canvas(plot_width=600, plot_height=600)
agg1 = canvas1.points(ds_df, 'x', 'y', ds.sum('prob'))
img1 = tf.shade(agg1, cmap=inferno)
axes[0].imshow(img1.to_pil())
axes[0].set_title('X-Y Projection (Aggregated by Probability)')
axes[0].set_xlabel('Dimension 1')
axes[0].set_ylabel('Dimension 2')
axes[0].set_xticks([])
axes[0].set_yticks([])

# X-Z projection
canvas2 = ds.Canvas(plot_width=600, plot_height=600)
agg2 = canvas2.points(ds_df, 'x', 'z', ds.sum('prob'))
img2 = tf.shade(agg2, cmap=inferno)
axes[1].imshow(img2.to_pil())
axes[1].set_title('X-Z Projection (Aggregated by Probability)')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 3')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Y-Z projection
canvas3 = ds.Canvas(plot_width=600, plot_height=600)
agg3 = canvas3.points(ds_df, 'y', 'z', ds.sum('prob'))
img3 = tf.shade(agg3, cmap=inferno)
axes[2].imshow(img3.to_pil())
axes[2].set_title('Y-Z Projection (Aggregated by Probability)')
axes[2].set_xlabel('Dimension 2')
axes[2].set_ylabel('Dimension 3')
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
plt.suptitle('Belief States Visualized with Datashader (Automatic Aggregation)', y=1.05, fontsize=16)
plt.show()

# %% 
# now do the same for the predictions
# Check if best_results is None before trying to visualize predictions
if best_results is not None and 'predictions' in best_results:
    # Convert predictions to numpy array for datashader
    predictions = best_results['predictions'].reshape(-1, 3)
    predictions_np = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    
    # Create DataFrame for datashader
    pred_df = pd.DataFrame({
        'x': predictions_np[:, 0],
        'y': predictions_np[:, 1],
        'z': predictions_np[:, 2],
        'weight': np.ones(len(predictions_np))
    })
    
    print("Creating datashader visualization for predictions...")
    
    # Set up the figure with three 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # X-Y projection (top-down view)
    canvas1 = ds.Canvas(plot_width=600, plot_height=600)
    agg1 = canvas1.points(pred_df, 'x', 'y', ds.count())
    img1 = tf.shade(agg1, cmap=inferno)
    axes[0].imshow(img1.to_pil())
    axes[0].set_title('X-Y Projection of Predictions')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # X-Z projection
    canvas2 = ds.Canvas(plot_width=600, plot_height=600)
    agg2 = canvas2.points(pred_df, 'x', 'z', ds.count())
    img2 = tf.shade(agg2, cmap=inferno)
    axes[1].imshow(img2.to_pil())
    axes[1].set_title('X-Z Projection of Predictions')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 3')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Y-Z projection
    canvas3 = ds.Canvas(plot_width=600, plot_height=600)
    agg3 = canvas3.points(pred_df, 'y', 'z', ds.count())
    img3 = tf.shade(agg3, cmap=inferno)
    axes[2].imshow(img3.to_pil())
    axes[2].set_title('Y-Z Projection of Predictions')
    axes[2].set_xlabel('Dimension 2')
    axes[2].set_ylabel('Dimension 3')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle('Predictions Visualized with Datashader (Automatic Aggregation)', y=1.05, fontsize=16)
    plt.show()
    
    # Compare beliefs and predictions in the same plot
    print("Creating comparison visualization for beliefs and predictions...")
    
    # Set up the figure with three 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Import proper colormaps from matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define a few reliable colormaps for use with datashader
    viridis_cmap = plt.cm.viridis
    plasma_cmap = plt.cm.plasma
    blues_cmap = plt.cm.Blues
    reds_cmap = plt.cm.Reds
    
    # X-Y projection (top-down view)
    canvas1 = ds.Canvas(plot_width=600, plot_height=600)
    # Plot beliefs with blue color
    agg1_beliefs = canvas1.points(ds_df, 'x', 'y', ds.sum('prob'))
    img1_beliefs = tf.shade(agg1_beliefs, cmap=blues_cmap, alpha=0.7, min_alpha=0.2)
    # Plot predictions with red color
    agg1_preds = canvas1.points(pred_df, 'x', 'y', ds.count())
    img1_preds = tf.shade(agg1_preds, cmap=reds_cmap, alpha=0.7, min_alpha=0.2)
    # Combine images
    img1_combined = tf.stack(img1_beliefs, img1_preds)
    axes[0].imshow(img1_combined.to_pil())
    axes[0].set_title('X-Y Projection (Blue: Beliefs, Red: Predictions)')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # X-Z projection
    canvas2 = ds.Canvas(plot_width=600, plot_height=600)
    agg2_beliefs = canvas2.points(ds_df, 'x', 'z', ds.sum('prob'))
    img2_beliefs = tf.shade(agg2_beliefs, cmap=blues_cmap, alpha=0.7, min_alpha=0.2)
    agg2_preds = canvas2.points(pred_df, 'x', 'z', ds.count())
    img2_preds = tf.shade(agg2_preds, cmap=reds_cmap, alpha=0.7, min_alpha=0.2)
    img2_combined = tf.stack(img2_beliefs, img2_preds)
    axes[1].imshow(img2_combined.to_pil())
    axes[1].set_title('X-Z Projection (Blue: Beliefs, Red: Predictions)')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 3')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Y-Z projection
    canvas3 = ds.Canvas(plot_width=600, plot_height=600)
    agg3_beliefs = canvas3.points(ds_df, 'y', 'z', ds.sum('prob'))
    img3_beliefs = tf.shade(agg3_beliefs, cmap=blues_cmap, alpha=0.7, min_alpha=0.2)
    agg3_preds = canvas3.points(pred_df, 'y', 'z', ds.count())
    img3_preds = tf.shade(agg3_preds, cmap=reds_cmap, alpha=0.7, min_alpha=0.2)
    img3_combined = tf.stack(img3_beliefs, img3_preds)
    axes[2].imshow(img3_combined.to_pil())
    axes[2].set_title('Y-Z Projection (Blue: Beliefs, Red: Predictions)')
    axes[2].set_xlabel('Dimension 2')
    axes[2].set_ylabel('Dimension 3')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle('Comparison of Beliefs and Predictions', y=1.05, fontsize=16)
    plt.show()
else:
    print("Cannot visualize predictions: best_results is None or doesn't contain predictions")

# %%
# Create a figure with two rows: ground truth (beliefs) and predictions
if best_results is not None and 'predictions' in best_results:
    # Process beliefs data
    nn_beliefs_np = nn_beliefs_reshaped.detach().cpu().numpy()
    nn_probs_np = nn_probs_reshaped.detach().cpu().numpy()
    
    ds_df = pd.DataFrame({
        'x': nn_beliefs_np[:, 0],
        'y': nn_beliefs_np[:, 1],
        'z': nn_beliefs_np[:, 2],
        'prob': nn_probs_np
    })
    
    # Process predictions data
    predictions = best_results['predictions'].reshape(-1, 3)
    predictions_np = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    
    pred_df = pd.DataFrame({
        'x': predictions_np[:, 0],
        'y': predictions_np[:, 1],
        'z': predictions_np[:, 2],
        'weight': np.ones(len(predictions))
    })
    
    print("Creating comparison figure with ground truth and predictions...")
    
    # Create a 2x3 subplot grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Use consistent colormaps for beliefs and predictions
    belief_cmap = 'Blues'  # Colormap for beliefs (ground truth)
    pred_cmap = 'Reds'     # Colormap for predictions
    
    # Row 1: Ground Truth (Beliefs)
    # X-Y projection of beliefs
    canvas_b1 = ds.Canvas(plot_width=600, plot_height=600)
    agg_b1 = canvas_b1.points(ds_df, 'x', 'y', ds.sum('prob'))
    img_b1 = tf.shade(agg_b1, cmap=belief_cmap)
    axes[0, 0].imshow(img_b1.to_pil())
    axes[0, 0].set_title('Ground Truth: X-Y Projection')
    axes[0, 0].set_xlabel('Dimension 1')
    axes[0, 0].set_ylabel('Dimension 2')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    # X-Z projection of beliefs
    canvas_b2 = ds.Canvas(plot_width=600, plot_height=600)
    agg_b2 = canvas_b2.points(ds_df, 'x', 'z', ds.sum('prob'))
    img_b2 = tf.shade(agg_b2, cmap=belief_cmap)
    axes[0, 1].imshow(img_b2.to_pil())
    axes[0, 1].set_title('Ground Truth: X-Z Projection')
    axes[0, 1].set_xlabel('Dimension 1')
    axes[0, 1].set_ylabel('Dimension 3')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    # Y-Z projection of beliefs
    canvas_b3 = ds.Canvas(plot_width=600, plot_height=600)
    agg_b3 = canvas_b3.points(ds_df, 'y', 'z', ds.sum('prob'))
    img_b3 = tf.shade(agg_b3, cmap=belief_cmap)
    axes[0, 2].imshow(img_b3.to_pil())
    axes[0, 2].set_title('Ground Truth: Y-Z Projection')
    axes[0, 2].set_xlabel('Dimension 2')
    axes[0, 2].set_ylabel('Dimension 3')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    # Row 2: Predictions
    # X-Y projection of predictions
    canvas_p1 = ds.Canvas(plot_width=600, plot_height=600)
    agg_p1 = canvas_p1.points(pred_df, 'x', 'y', ds.sum())
    img_p1 = tf.shade(agg_p1, cmap=pred_cmap)
    axes[1, 0].imshow(img_p1.to_pil())
    axes[1, 0].set_title('Predictions: X-Y Projection')
    axes[1, 0].set_xlabel('Dimension 1')
    axes[1, 0].set_ylabel('Dimension 2')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    # X-Z projection of predictions
    canvas_p2 = ds.Canvas(plot_width=600, plot_height=600)
    agg_p2 = canvas_p2.points(pred_df, 'x', 'z', ds.count())
    img_p2 = tf.shade(agg_p2, cmap=pred_cmap)
    axes[1, 1].imshow(img_p2.to_pil())
    axes[1, 1].set_title('Predictions: X-Z Projection')
    axes[1, 1].set_xlabel('Dimension 1')
    axes[1, 1].set_ylabel('Dimension 3')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Y-Z projection of predictions
    canvas_p3 = ds.Canvas(plot_width=600, plot_height=600)
    agg_p3 = canvas_p3.points(pred_df, 'y', 'z', ds.count())
    img_p3 = tf.shade(agg_p3, cmap=pred_cmap)
    axes[1, 2].imshow(img_p3.to_pil())
    axes[1, 2].set_title('Predictions: Y-Z Projection')
    axes[1, 2].set_xlabel('Dimension 2')
    axes[1, 2].set_ylabel('Dimension 3')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle('Ground Truth vs Predictions Comparison', y=1.02, fontsize=16)
    plt.show()
else:
    print("Cannot create comparison visualization: best_results is None or doesn't contain predictions")

# %%
# Create a side-by-side comparison using proper colormaps (fixing any transparency issues)

ds_res = 175
pred_res = 400
if best_results is not None and 'predictions' in best_results:
    # Process beliefs data
    nn_beliefs_np = nn_beliefs_reshaped.detach().cpu().numpy()
    nn_probs_np = nn_probs_reshaped.detach().cpu().numpy()
    
    ds_df = pd.DataFrame({
        'x': nn_beliefs_np[:, 0],
        'y': nn_beliefs_np[:, 1],
        'z': nn_beliefs_np[:, 2],
        'prob': nn_probs_np
    })
    
    # Process predictions data
    predictions = best_results['predictions'].reshape(-1, 3)
    predictions_np = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    
    pred_df = pd.DataFrame({
        'x': predictions_np[:, 0],
        'y': predictions_np[:, 1],
        'z': predictions_np[:, 2],
        'weight': nn_probs_np
    })
    
    print("Creating safe side-by-side visualization...")
    
    # Use built-in colormaps with white for low values
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    
    # Define a colormap where low values are white
    white_to_blue = plt.cm.Reds
    white_to_red = plt.cm.Reds
    
    # X-Y projection
    # Ground truth (smaller canvas)
    canvas1_gt = ds.Canvas(plot_width=ds_res, plot_height=ds_res)
    agg1_gt = canvas1_gt.points(ds_df, 'x', 'y', ds.sum('prob'))
    img1_gt = tf.shade(agg1_gt, cmap=white_to_blue)
    
    # Predictions (larger canvas)
    canvas1_pred = ds.Canvas(plot_width=pred_res, plot_height=pred_res)
    agg1_pred = canvas1_pred.points(pred_df, 'x', 'y', ds.sum('weight'))
    img1_pred = tf.shade(agg1_pred, cmap=white_to_red)
    
    # Display in the main figure with clear labels
    axes[0, 0].imshow(img1_gt.to_pil())
    axes[0, 0].set_title('Ground Truth: X-Y Projection')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    axes[1, 0].imshow(img1_pred.to_pil())
    axes[1, 0].set_title('Predictions: X-Y Projection')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    # X-Z projection
    # Ground truth (smaller canvas)
    canvas2_gt = ds.Canvas(plot_width=ds_res, plot_height=ds_res)
    agg2_gt = canvas2_gt.points(ds_df, 'x', 'z', ds.sum('prob'))
    img2_gt = tf.shade(agg2_gt, cmap=white_to_blue)
    
    # Predictions (larger canvas)
    canvas2_pred = ds.Canvas(plot_width=pred_res, plot_height=pred_res)
    agg2_pred = canvas2_pred.points(pred_df, 'x', 'z', ds.sum('weight'))
    img2_pred = tf.shade(agg2_pred, cmap=white_to_red)
    
    axes[0, 1].imshow(img2_gt.to_pil())
    axes[0, 1].set_title('Ground Truth: X-Z Projection')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    axes[1, 1].imshow(img2_pred.to_pil())
    axes[1, 1].set_title('Predictions: X-Z Projection')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Y-Z projection
    # Ground truth (smaller canvas)
    canvas3_gt = ds.Canvas(plot_width=ds_res, plot_height=ds_res)
    agg3_gt = canvas3_gt.points(ds_df, 'y', 'z', ds.sum('prob'))
    img3_gt = tf.shade(agg3_gt, cmap=white_to_blue)
    
    # Predictions (larger canvas)
    canvas3_pred = ds.Canvas(plot_width=pred_res, plot_height=pred_res)
    agg3_pred = canvas3_pred.points(pred_df, 'y', 'z', ds.sum('weight'))
    img3_pred = tf.shade(agg3_pred, cmap=white_to_red)
    
    axes[0, 2].imshow(img3_gt.to_pil())
    axes[0, 2].set_title('Ground Truth: Y-Z Projection')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    axes[1, 2].imshow(img3_pred.to_pil())
    axes[1, 2].set_title('Predictions: Y-Z Projection')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle('Ground Truth vs Predictions Comparison', y=0.98, fontsize=16)
    plt.show()
else:
    print("Cannot create visualization: missing data")

# %%
# Manual aggregation without datashader - focused on Y-Z projections only
if best_results is not None and 'predictions' in best_results:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Process beliefs data
    nn_beliefs_np = nn_beliefs_reshaped.detach().cpu().numpy()
    nn_probs_np = nn_probs_reshaped.detach().cpu().numpy()
    
    # Process predictions data
    predictions = best_results['predictions'].reshape(-1, 3)
    predictions_np = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    
    # Define the number of bins for the manual aggregation
    n_bins = 200  # Adjust this for different resolution
    
    # Define manual binning functions
    def create_bins(data, n_bins):
        """Create evenly spaced bins for a data dimension"""
        min_val = np.min(data)
        max_val = np.max(data)
        # Add a small padding to ensure all data points are included
        padding = (max_val - min_val) * 0.01
        return np.linspace(min_val - padding, max_val + padding, n_bins + 1)
    
    def manual_aggregation(y_vals, z_vals, weights, n_bins):
        """Manually aggregate data into bins and sum weights"""
        # Create bins for y and z dimensions
        y_bins = create_bins(y_vals, n_bins)
        z_bins = create_bins(z_vals, n_bins)
        
        # Initialize aggregation grid
        grid = np.zeros((n_bins, n_bins))
        
        # Assign each point to its bin and aggregate
        for i in range(len(y_vals)):
            y_bin = np.digitize(y_vals[i], y_bins) - 1
            z_bin = np.digitize(z_vals[i], z_bins) - 1
            
            # Ensure the bin indices are valid (within bounds)
            if 0 <= y_bin < n_bins and 0 <= z_bin < n_bins:
                grid[z_bin, y_bin] += weights[i]  # Note: rows=z, cols=y for proper orientation
        
        return grid, y_bins, z_bins
    
    # Create figure for Y-Z projections only
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Manual aggregation for beliefs (ground truth)
    y_gt = nn_beliefs_np[:, 1]  # Y dimension
    z_gt = nn_beliefs_np[:, 2]  # Z dimension
    gt_grid, gt_y_bins, gt_z_bins = manual_aggregation(y_gt, z_gt, nn_probs_np, n_bins)
    
    # Manual aggregation for predictions
    y_pred = predictions_np[:, 1]  # Y dimension
    z_pred = predictions_np[:, 2]  # Z dimension
    # Use the same probability weights for predictions as well
    pred_grid, pred_y_bins, pred_z_bins = manual_aggregation(y_pred, z_pred, nn_probs_np, n_bins)
    
    # Create a custom colormap with proper handling of low values
    cmap_blues = plt.cm.Blues
    cmap_reds = plt.cm.Reds
    
    # Create extent parameters for proper axis scaling
    gt_extent = [gt_y_bins[0], gt_y_bins[-1], gt_z_bins[0], gt_z_bins[-1]]
    pred_extent = [pred_y_bins[0], pred_y_bins[-1], pred_z_bins[0], pred_z_bins[-1]]
    
    # Plot ground truth
    im0 = axes[0].imshow(gt_grid, origin='lower', aspect='auto', 
                         extent=gt_extent, cmap=cmap_blues)
    axes[0].set_title('Ground Truth: Y-Z Projection\n(Manual Aggregation)')
    axes[0].set_xlabel('Dimension 2 (Y)')
    axes[0].set_ylabel('Dimension 3 (Z)')
    fig.colorbar(im0, ax=axes[0], label='Sum of Probabilities')
    
    # Plot predictions
    im1 = axes[1].imshow(pred_grid, origin='lower', aspect='auto', 
                         extent=pred_extent, cmap=cmap_reds)
    axes[1].set_title('Predictions: Y-Z Projection\n(Manual Aggregation)')
    axes[1].set_xlabel('Dimension 2 (Y)')
    axes[1].set_ylabel('Dimension 3 (Z)')
    fig.colorbar(im1, ax=axes[1], label='Sum of Probabilities')
    
    # Add some statistics information
    gt_stats = f"Max: {gt_grid.max():.3f}, Sum: {gt_grid.sum():.3f}\nNon-zero bins: {np.count_nonzero(gt_grid)}/{n_bins*n_bins}"
    pred_stats = f"Max: {pred_grid.max():.3f}, Sum: {pred_grid.sum():.3f}\nNon-zero bins: {np.count_nonzero(pred_grid)}/{n_bins*n_bins}"
    
    axes[0].text(0.05, 0.95, gt_stats, transform=axes[0].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1].text(0.05, 0.95, pred_stats, transform=axes[1].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.suptitle('Y-Z Projections with Manual Aggregation', y=1.05)
    plt.show()
    
    # Compare different bin resolutions for ground truth Y-Z projection
    bin_sizes = [20, 50, 100, 200]
    fig, axes = plt.subplots(1, len(bin_sizes), figsize=(15, 4))
    
    for i, bins in enumerate(bin_sizes):
        grid, y_bins, z_bins = manual_aggregation(y_gt, z_gt, nn_probs_np, bins)
        extent = [y_bins[0], y_bins[-1], z_bins[0], z_bins[-1]]
        
        im = axes[i].imshow(grid, origin='lower', aspect='auto', 
                           extent=extent, cmap=cmap_blues)
        axes[i].set_title(f'{bins}×{bins} bins')
        axes[i].set_xlabel('Y')
        axes[i].set_ylabel('Z')
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.suptitle('Effect of Bin Resolution on Manual Aggregation', y=1.05)
    plt.show()
else:
    print("Cannot create manual aggregation visualization: missing data")

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Helper: Manual aggregation for multiple arrays
# =============================================================================
def manual_aggregate_dict(x, y, data_dict, x_bins, y_bins):
    """
    Aggregates arrays (in data_dict) into bins defined by x_bins (horizontal axis)
    and y_bins (vertical axis).
    
    Parameters:
        x, y: arrays of coordinates (used for binning; here x is horizontal, y is vertical)
        data_dict: dictionary mapping names to arrays (each of same length as x, y)
        x_bins, y_bins: arrays of bin edges for horizontal and vertical axes.
    
    Returns:
        A dictionary mapping each key to a 2D aggregated array.
    """
    grid_shape = (len(y_bins) - 1, len(x_bins) - 1)
    aggs = {key: np.zeros(grid_shape) for key in data_dict}
    
    # Find bin indices for each point
    x_idx = np.digitize(x, x_bins) - 1  # horizontal index
    y_idx = np.digitize(y, y_bins) - 1  # vertical index

    for xi, yi, idx in zip(x_idx, y_idx, range(len(x))):
        if 0 <= xi < grid_shape[1] and 0 <= yi < grid_shape[0]:
            for key, arr in data_dict.items():
                aggs[key][yi, xi] += arr[idx]
    return aggs

# =============================================================================
# Helper: Create an image from aggregated values with weighted average RGB and transparency
# =============================================================================
def create_manual_image(aggs, epsilon=1e-8, alpha_scale=300.0):
    """
    Computes the weighted average RGB per pixel and sets the alpha channel based on
    the aggregated probability.
    
    Parameters:
        aggs: dictionary with keys 'prob', 'R_weight', 'G_weight', 'B_weight'
        epsilon: small value to avoid division by zero
        alpha_scale: factor to scale alpha (transparency); values > 1 make points more visible
    
    Returns:
        img: an (H, W, 4) uint8 image array (RGBA)
        max_prob: maximum aggregated probability (for metric reporting)
        nonzero_count: number of pixels with nonzero probability
    """
    prob = aggs['prob']
    Rw = aggs['R_weight']
    Gw = aggs['G_weight']
    Bw = aggs['B_weight']
    H, W = prob.shape
    
    # Compute weighted average colors for pixels that have data.
    avg_R = np.zeros_like(prob, dtype=float)
    avg_G = np.zeros_like(prob, dtype=float)
    avg_B = np.zeros_like(prob, dtype=float)
    mask = prob > epsilon
    avg_R[mask] = Rw[mask] / prob[mask]
    avg_G[mask] = Gw[mask] / prob[mask]
    avg_B[mask] = Bw[mask] / prob[mask]
    
    # Scale normalized average colors to [0, 255]
    avg_R_scaled = (avg_R * 255).astype(np.uint8)
    avg_G_scaled = (avg_G * 255).astype(np.uint8)
    avg_B_scaled = (avg_B * 255).astype(np.uint8)
    
    # Compute alpha channel with scaling
    max_prob = prob.max() if prob.max() > 0 else 1
    alpha = np.zeros_like(prob, dtype=float)
    alpha[mask] = np.clip(alpha_scale * prob[mask] / max_prob, 0, 1) * 255
    alpha_scaled = alpha.astype(np.uint8)
    
    # Combine channels into an RGBA image.
    img = np.stack([avg_R_scaled, avg_G_scaled, avg_B_scaled, alpha_scaled], axis=-1)
    return img, max_prob, np.count_nonzero(prob)
# =============================================================================
# Set up manual aggregation for beliefs
# =============================================================================
# For our Y–Z projection, we use:
#   Horizontal axis: 'y' from ds_df
#   Vertical axis: 'z' from ds_df
#
# Assume ds_df has been created as in your code and includes the following columns:
#   - 'prob': the probability weight for each belief point
#   - 'R_weight': normalized x * prob
#   - 'G_weight': normalized y * prob
#   - 'B_weight': normalized sqrt(x²+y²) * prob

# Define number of bins (resolution)
n_bins = 201

# Use the min/max of y and z defined earlier (from your ds_df)
# (They were computed as:)
# min_y = min(ds_df['y'].min(), pred_df['y'].min())
# max_y = max(ds_df['y'].max(), pred_df['y'].max())
# min_z = min(ds_df['z'].min(), pred_df['z'].min())
# max_z = max(ds_df['z'].max(), pred_df['z'].max())

# Create bin edges for the Y (horizontal) and Z (vertical) axes.
y_bins = np.linspace(min_y, max_y, n_bins + 1)
z_bins = np.linspace(min_z, max_z, n_bins + 1)

# Aggregate over the points. For Y–Z, use ds_df['y'] and ds_df['z'] as coordinates.
data_dict = {
    'prob': ds_df['prob'].values,
    'R_weight': ds_df['R_weight'].values,
    'G_weight': ds_df['G_weight'].values,
    'B_weight': ds_df['B_weight'].values
}

aggs_beliefs = manual_aggregate_dict(ds_df['y'].values, ds_df['z'].values, data_dict, y_bins, z_bins)
#%%
pred_data_dict = {
    'prob': pred_df['weight'].values,
    'R_weight': pred_df['R_weight'].values,
    'G_weight': pred_df['G_weight'].values,
    'B_weight': pred_df['B_weight'].values
}
#%%
aggs_predictions = manual_aggregate_dict(pred_df['y'].values, pred_df['z'].values, pred_data_dict, y_bins, z_bins)
#%%
# Create the RGBA image from the aggregated values.
img_beliefs_manual, max_prob_manual, nonzero_manual = create_manual_image(aggs_beliefs)
img_predictions_manual, max_prob_manual, nonzero_manual = create_manual_image(aggs_predictions)

# =============================================================================
# Visualize the manually aggregated image
# =============================================================================
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
extent = tuple([y_bins[0], y_bins[-1], z_bins[0], z_bins[-1]])  # for proper axis scaling

# First subplot - beliefs
ax1.imshow(img_beliefs_manual, origin='lower', extent=extent)
ax1.set_title(f'Beliefs')
#\nMax Prob: {max_prob_manual:.3f}\nNon-zero bins: {nonzero_manual}')

ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis('off')

# Second subplot - predictions
ax2.imshow(img_predictions_manual, origin='lower', extent=extent)
ax2.set_title(f'Predictions')
#\nMax Prob: {max_prob_manual:.3f}\nNon-zero bins: {nonzero_manual}')

ax2.set_xticks([])
ax2.set_yticks([])
ax2.axis('off')

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf

# -----------------------------------------------------------------------------
# Custom Shader Function
# -----------------------------------------------------------------------------
def custom_shader(agg, epsilon=1e-8, alpha_scale=300.0):
    """
    Convert a datashader aggregation (an xarray.Dataset) into an RGBA image,
    computing weighted average RGB values and applying a scaled alpha based on 
    the aggregated probability.

    Parameters:
        agg         : xarray.Dataset with keys 'R_weight', 'G_weight', 'B_weight', 'prob_sum'
        epsilon     : Small constant to avoid division by zero.
        alpha_scale : Scaling factor for alpha; values >1 will increase opacity for low-probability pixels.
    
    Returns:
        An Image object (from datashader.transfer_functions) that can be converted to a PIL image.
    """
    # Extract aggregated arrays (assumed to be 2D with dimensions "y" and "z")
    Rw   = agg['R_weight'].values
    Gw   = agg['G_weight'].values
    Bw   = agg['B_weight'].values
    prob = agg['prob_sum'].values

    # Create a mask for pixels with nonzero probability
    mask = prob > epsilon

    # Compute weighted averages for each color channel (default 0 when no data)
    avg_R = np.zeros_like(prob, dtype=float)
    avg_G = np.zeros_like(prob, dtype=float)
    avg_B = np.zeros_like(prob, dtype=float)
    avg_R[mask] = Rw[mask] / prob[mask]
    avg_G[mask] = Gw[mask] / prob[mask]
    avg_B[mask] = Bw[mask] / prob[mask]

    # Scale the color channels to 0-255 (assuming colors were normalized to [0, 1])
    avg_R_scaled = (avg_R * 255).astype(np.uint8)
    avg_G_scaled = (avg_G * 255).astype(np.uint8)
    avg_B_scaled = (avg_B * 255).astype(np.uint8)

    # Compute alpha channel: scale the aggregated probability relative to the maximum
    max_prob = prob.max() if prob.max() > 0 else 1
    alpha = np.zeros_like(prob, dtype=float)
    alpha[mask] = np.clip(alpha_scale * prob[mask] / max_prob, 0, 1) * 255
    alpha_scaled = alpha.astype(np.uint8)

    # Stack the channels into an RGBA image array; shape will be (height, width, 4)
    img_array = np.stack([avg_R_scaled, avg_G_scaled, avg_B_scaled, alpha_scaled], axis=-1)

    # Create a coordinate dictionary using the existing "y" and "z" coordinates,
    # and define a "band" coordinate for the channels.
    coords = {
        "y": agg.coords["y"].values,
        "z": agg.coords["z"].values,
        "band": ["R", "G", "B", "A"]
    }
    dims = ("y", "z", "band")

    # Return as a datashader Image object
    return tf.Image(img_array, coords=coords, dims=dims)

# -----------------------------------------------------------------------------
# Define a Canvas and Aggregate Data with a Summary Reduction
# -----------------------------------------------------------------------------
# Assume that you already have two pandas DataFrames: 
#   - ds_df for beliefs with columns: 'y', 'z', 'prob', 'R_weight', 'G_weight', 'B_weight'
#   - pred_df for predictions with columns: 'y', 'z', 'weight', 'R_weight', 'G_weight', 'B_weight'
#
# Note: In ds_df, the key 'prob' is used; in the aggregation we rename it to 'prob_sum'
#       (for predictions, we sum 'weight').

# Determine the ranges for the Y-Z projection (you might have computed these earlier)
min_y = ds_df['y'].min()
max_y = ds_df['y'].max()
min_z = ds_df['z'].min()
max_z = ds_df['z'].max()

# Create a datashader canvas (you can adjust the resolution as desired)
canvas = ds.Canvas(plot_width=300, plot_height=300, x_range=(min_y, max_y), y_range=(min_z, max_z))

# Aggregate beliefs using a summary reduction
agg_beliefs = canvas.points(
    ds_df,
    x='y', 
    y='z',
    agg=ds.reductions.summary(
        R_weight=ds.reductions.sum('R_weight'),
        G_weight=ds.reductions.sum('G_weight'),
        B_weight=ds.reductions.sum('B_weight'),
        prob_sum=ds.reductions.sum('prob')
    )
)

# Aggregate predictions (here we use 'weight' for probability)
agg_predictions = canvas.points(
    pred_df,
    x='y',
    y='z',
    agg=ds.reductions.summary(
        R_weight=ds.reductions.sum('R_weight'),
        G_weight=ds.reductions.sum('G_weight'),
        B_weight=ds.reductions.sum('B_weight'),
        prob_sum=ds.reductions.sum('weight')
    )
)

# -----------------------------------------------------------------------------
# Generate Images with the Custom Shader and Visualize
# -----------------------------------------------------------------------------
img_beliefs = custom_shader(agg_beliefs, alpha_scale=120000.0)
img_predictions = custom_shader(agg_predictions, alpha_scale=8000.0)

# Plot side-by-side comparisons
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(img_beliefs.to_pil(), origin='lower')
ax1.set_title('Beliefs (Custom Shader)')
ax1.axis('off')

ax2.imshow(img_predictions.to_pil(), origin='lower')
ax2.set_title('Predictions (Custom Shader)')
ax2.axis('off')

plt.tight_layout()
plt.show()
# %%
# %%
import scipy.ndimage

def smooth_image(img_array, sigma=1.0):
    # Apply Gaussian blur to each channel separately.
    smoothed = np.empty_like(img_array)
    for channel in range(img_array.shape[-1]):
        smoothed[..., channel] = scipy.ndimage.gaussian_filter(img_array[..., channel], sigma=sigma)
    return smoothed

# Generate the image using your custom shader
img_beliefs = custom_shader(agg_beliefs, alpha_scale=18000.0)
img_predictions = custom_shader(agg_predictions, alpha_scale=13000.0)

# Convert to numpy arrays (if not already) and smooth them
img_beliefs_array = img_beliefs.data  # This is the underlying numpy array
img_predictions_array = img_predictions.data

# Apply smoothing (adjust sigma as needed)
img_beliefs_smoothed = smooth_image(img_beliefs_array, sigma=1)
img_predictions_smoothed = smooth_image(img_predictions_array, sigma=.5)

# Wrap them back into a datashader Image (optional, or convert directly to PIL)
smoothed_beliefs = tf.Image(img_beliefs_smoothed, coords=img_beliefs.coords, dims=img_beliefs.dims)
smoothed_predictions = tf.Image(img_predictions_smoothed, coords=img_predictions.coords, dims=img_predictions.dims)

# Visualize the smoothed images
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(smoothed_beliefs.to_pil(), origin='lower')
ax1.set_title('Beliefs (Smoothed)')
ax1.axis('off')

ax2.imshow(smoothed_predictions.to_pil(), origin='lower')
ax2.set_title('Predictions (Smoothed)')
ax2.axis('off')

plt.tight_layout()
plt.show()
# %%
import numpy as np
import plotly.graph_objs as go

# -----------------------------------------------------------------------------
# Compute Normalized RGB Components from Belief Coordinates
# -----------------------------------------------------------------------------
# Extract coordinates from your beliefs array.
x = nn_beliefs_np[:, 0]   # used for R
y = nn_beliefs_np[:, 1]   # used for G (and as x-axis for plotting)
z = nn_beliefs_np[:, 2]   # used for plotting as y-axis (for the Y-Z projection)

# Compute B as the distance in the xy-plane.
B_val = np.sqrt(x**2 + y**2)

# Compute global min/max for each channel.
epsilon = 1e-8
min_R, max_R = x.min(), x.max()
min_G, max_G = y.min(), y.max()
min_B, max_B = B_val.min(), B_val.max()

# Normalize each channel to [0,1]
R_norm = (x - min_R) / (max_R - min_R + epsilon)
G_norm = (y - min_G) / (max_G - min_G + epsilon)
B_norm = (B_val - min_B) / (max_B - min_B + epsilon)

# Convert normalized values to 0-255 integers.
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
    
    Parameters:
        prob      : float or numpy array (0-1)
        scale     : Controls the nonlinearity; higher values boost low probabilities more.
        min_alpha : Minimum alpha value (float in [0,1]).
    
    Returns:
        A numpy array of alpha values in the range [0, 1].
    """
    # log1p helps to gently boost low values (ensuring 0 stays 0)
    transformed = np.log1p(prob * scale) / np.log1p(scale)
    alpha = min_alpha + (1 - min_alpha) * transformed
    return np.clip(alpha, 0, 1)

alpha_float = compute_alpha_float(nn_probs_np, scale=1, min_alpha=0.15)

# -----------------------------------------------------------------------------
# Create RGBA Strings with Floating Point Alpha
# -----------------------------------------------------------------------------
colors = [f"rgba({r},{g},{b},{alpha:.2f})" 
          for r, g, b, alpha in zip(R_int, G_int, B_int, alpha_float)]

from epsilon_transformers.visualization.plots import _project_to_simplex

# -----------------------------------------------------------------------------
# Create side-by-side Plotly Scattergl Plots with clean white background
# -----------------------------------------------------------------------------
# Create subplot figure
fig = go.Figure()

# Beliefs plot
x_, y_ = _project_to_simplex(nn_beliefs.reshape(-1, 3))
beliefs_trace = go.Scattergl(
    x=x_,
    y=y_,
    mode='markers',
    marker=dict(
        color=colors,
        size=4#2
    ),
    name="Belief States"
)
fig.add_trace(beliefs_trace)

# Predictions plot
x = pred_df['x'].values
y = pred_df['y'].values
z = pred_df['z'].values

# Compute alpha values for predictions
pred_probs_np = pred_df['weight'].values
alpha_float_pred = compute_alpha_float(pred_probs_np, scale=1, min_alpha=0.15)

# Create RGBA colors for predictions
pred_colors = [f"rgba({r},{g},{b},{alpha:.2f})" 
               for r, g, b, alpha in zip(R_int, G_int, B_int, alpha_float_pred)]

x_, y_ = _project_to_simplex(pred_df[['x', 'y', 'z']].values)

# Create the scatter plot for predictions
pred_trace = go.Scattergl(
    x=x_,
    y=y_,
    mode='markers',
    marker=dict(
        color=pred_colors,
        size=4#2
    ),
    name="Predictions",
    visible=False
)
fig.add_trace(pred_trace)

# Create clean white background with no grid, labels or numbers
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title=""
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title=""
    ),
    updatemenus=[dict(
        type="buttons",
        direction="right",
        buttons=[
            dict(
                label="Beliefs",
                method="update",
                args=[{"visible": [True, False]}]
            ),
            dict(
                label="Predictions",
                method="update",
                args=[{"visible": [False, True]}]
            )
        ],
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.1,
        yanchor="top"
    )]
)

fig.show()


# %%
