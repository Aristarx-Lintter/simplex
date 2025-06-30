#%% [markdown]
# # GPTs Analysis Runner
#
# This script runs the core regression analysis for specified process/model combinations,
# saves detailed results (metrics, predictions, ground truth, weights, dimensionality info)
# to pickle files for later plotting. It uses K-Fold cross-validation and handles
# activation deduplication based on input prefixes.
#
# **NOTE:** This version has all try-except blocks removed for explicit error reporting.
# Ensure all dependencies and helper functions are correctly imported and defined.

#%% [markdown]
# ## Imports

#%%
import torch
import os
import pickle
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm

# Make sure these imports point to the correct location of your functions
# Using the exact imports provided by the user
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
# Assuming ActivationExtractor and TRANSFORMER_ACTIVATION_KEYS are needed by minimal_layerwise_regression or its callees
from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
# Assuming RegressionAnalyzer and RCOND_SWEEP_LIST might be needed internally or passed
from scripts.activation_analysis.config import RCOND_SWEEP_LIST

# The core function for running regression with KFold
from scripts.activation_analysis.regression import run_activation_to_beliefs_regression_kf


#%%
def compute_kfold_split(flat_probs, n_splits=5, random_state=42):
    from sklearn.model_selection import KFold
    import numpy as np

    # If flat_probs is a PyTorch tensor, convert it to numpy
    if isinstance(flat_probs, torch.Tensor):
        flat_probs = flat_probs.cpu().detach().numpy()
    
    # Create position indices
    all_positions = np.arange(len(flat_probs))
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    
    # Return the KFold object and positions
    # kf is a list of tuples, each tuple contains two lists: the indices of the training set and the indices of the test set
    return kf, all_positions

def find_duplicate_prefixes(nn_inputs):
    """
    Find duplicate prefixes in the input sequences and return their indices.
    
    Args:
        nn_inputs: Tensor of shape (batch_size, seq_len) containing token sequences
        
    Returns:
        Dictionary mapping each unique prefix tuple to a list of (seq_idx, pos) tuples
    """
    batch_size, seq_len = nn_inputs.shape
    prefix_to_indices = {}  # (prefix tuple) -> list of (seq_idx, pos) tuples
    
    # Process each sequence and position
    for seq_idx in range(batch_size):
        seq = nn_inputs[seq_idx]
        
        for pos in range(seq_len):
            # Get the prefix up to this position
            prefix = tuple(seq[:pos+1].cpu().numpy().tolist())
            
            # Add this occurrence to our mapping
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append((seq_idx, pos))
    
    return prefix_to_indices

def combine_duplicate_data(prefix_to_indices, activations, nn_probs, belief_states, debug=False, tolerance=1e-6):
    """
    Combine data for duplicate prefixes by summing probabilities.
    
    Args:
        prefix_to_indices: Dictionary mapping prefixes to lists of (seq_idx, pos) tuples
        activations: Tensor of shape (batch_size, seq_len, d_model)
        nn_probs: Tensor of shape (batch_size, seq_len)
        belief_states: Tensor of shape (batch_size, seq_len, belief_dim)
        debug: Whether to print debug information
        tolerance: Tolerance for activation differences
        
    Returns:
        Tuple of (unique_activations, summed_probs, unique_beliefs, unique_prefixes)
    """
    # Dictionary to store unique activations and summed probabilities
    unique_data = {}  # (prefix tuple) -> (activation, summed_prob, belief_state, count)
    
    # Debug information
    if debug:
        activation_diffs = {}  # (prefix tuple) -> max difference observed
        inconsistent_prefixes = []
    
    # Process each unique prefix
    for prefix, indices_list in prefix_to_indices.items():
        first_seq_idx, first_pos = indices_list[0]
        act = activations[first_seq_idx, first_pos]
        prob = nn_probs[first_seq_idx, first_pos]
        belief = belief_states[first_seq_idx, first_pos]
        count = 1
        
        # Process additional occurrences of this prefix
        for seq_idx, pos in indices_list[1:]:
            current_act = activations[seq_idx, pos]
            current_prob = nn_probs[seq_idx, pos]
            
            if debug:
                # Check if activations are the same
                diff = torch.max(torch.abs(act - current_act)).item()
                
                if diff > tolerance:
                    if prefix not in inconsistent_prefixes:
                        inconsistent_prefixes.append(prefix)
                    
                    current_max_diff = activation_diffs.get(prefix, 0)
                    activation_diffs[prefix] = max(current_max_diff, diff)
            
            # Sum the probabilities
            prob += current_prob
            count += 1
        
        # Store the combined data
        unique_data[prefix] = (act, prob, belief, count)
    
    # Print debug info if requested
    if debug:
        if inconsistent_prefixes:
            print(f"WARNING: Found {len(inconsistent_prefixes)} prefixes with inconsistent activations!")
            print(f"Max differences observed:")
            for prefix in sorted(inconsistent_prefixes, key=lambda p: activation_diffs[p], reverse=True)[:10]:
                print(f"  Prefix {prefix}: max diff = {activation_diffs[prefix]}, seen {unique_data[prefix][3]} times")
        else:
            print("All prefixes have consistent activations (within tolerance).")
            
        total_saved = sum(unique_data[p][3] - 1 for p in unique_data)
        total_items = sum(len(indices) for indices in prefix_to_indices.values())
        print(f"Deduplication saved {total_saved} activations out of {total_items}")
    
    # Convert to tensors
    unique_prefixes = list(unique_data.keys())
    unique_activations = torch.stack([unique_data[p][0] for p in unique_prefixes])
    summed_probs = torch.tensor([unique_data[p][1] for p in unique_prefixes])
    unique_beliefs = torch.stack([unique_data[p][2] for p in unique_prefixes])
    
    return unique_activations, summed_probs, unique_beliefs, unique_prefixes

def deduplicate_tensor(prefix_to_indices, tensor, aggregation_fn=None, debug=False, tolerance=1e-6):
    """
    Deduplicate a single tensor based on prefix indices.
    
    Args:
        prefix_to_indices: Dictionary mapping prefixes to lists of (seq_idx, pos) tuples
        tensor: Tensor of shape (batch_size, seq_len, ...) to deduplicate
        aggregation_fn: Function to aggregate duplicate values (default: take first occurrence)
                        Should accept a list of tensor values and return a single value
        debug: Whether to print debug information
        tolerance: Tolerance for tensor value differences
        
    Returns:
        Tuple of (unique_tensor_values, unique_prefixes)
    """
    # Dictionary to store unique tensor values
    unique_data = {}  # (prefix tuple) -> (tensor_value, count)
    
    # Debug information
    if debug:
        value_diffs = {}  # (prefix tuple) -> max difference observed
        inconsistent_prefixes = []
    
    # Process each unique prefix
    for prefix, indices_list in prefix_to_indices.items():
        first_seq_idx, first_pos = indices_list[0]
        value = tensor[first_seq_idx, first_pos]
        count = 1
        
        # If we have an aggregation function and multiple occurrences, prepare to aggregate
        if aggregation_fn is not None and len(indices_list) > 1:
            values = [value]
            
            # Collect all values for this prefix
            for seq_idx, pos in indices_list[1:]:
                current_value = tensor[seq_idx, pos]
                values.append(current_value)
                
                if debug:
                    # Check if values are the same
                    diff = torch.max(torch.abs(value - current_value)).item()
                    
                    if diff > tolerance:
                        if prefix not in inconsistent_prefixes:
                            inconsistent_prefixes.append(prefix)
                        
                        current_max_diff = value_diffs.get(prefix, 0)
                        value_diffs[prefix] = max(current_max_diff, diff)
                
                count += 1
            
            # Aggregate the values
            value = aggregation_fn(values)
        
        # Store the unique value
        unique_data[prefix] = (value, count)
    
    # Print debug info if requested
    if debug:
        if inconsistent_prefixes:
            print(f"WARNING: Found {len(inconsistent_prefixes)} prefixes with inconsistent values!")
            print(f"Max differences observed:")
            for prefix in sorted(inconsistent_prefixes, key=lambda p: value_diffs[p], reverse=True)[:10]:
                print(f"  Prefix {prefix}: max diff = {value_diffs[prefix]}, seen {unique_data[prefix][1]} times")
        else:
            print("All prefixes have consistent values (within tolerance).")
            
        total_saved = sum(unique_data[p][1] - 1 for p in unique_data)
        total_items = sum(len(indices) for indices in prefix_to_indices.values())
        print(f"Deduplication saved {total_saved} items out of {total_items}")
    
    # Convert to tensor
    unique_prefixes = list(unique_data.keys())
    unique_values = torch.stack([unique_data[p][0] for p in unique_prefixes])
    
    return unique_values, unique_prefixes


def get_nn_type(run_id):
    if 'GRU' in run_id or 'LSTM' in run_id or 'RNN' in run_id or 're' in run_id:
        return "RNN"
    else:
        return "transformer"

def prepare_msp_data_for_regression(
    sweep_id: str,
    run_id: str,
    checkpoint_path: str,
    device: str = "cpu",
):
    """
    Prepare MSP data for regression analysis
    
    Args:
        sweep_id: The sweep ID to load from
        run_id: The run ID to load from
        checkpoint_path: Path to the checkpoint file
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Tuple containing (model, run_config, nn_inputs, nn_beliefs, nn_probs)
    """
    # Load model and prepare data
    s3_loader = S3ModelLoader(use_company_credentials=True)
    model, run_config = s3_loader.load_checkpoint(
        sweep_id, run_id, checkpoint_path, device=device
    )
    
    # Prepare belief data
    n_ctx = run_config["model_config"]["n_ctx"]
    run_config["n_ctx"] = n_ctx
    nn_inputs, nn_beliefs, _, nn_probs, _ = prepare_msp_data(
        run_config, run_config["process_config"]
    )
    
    return model, run_config, nn_inputs, nn_beliefs, nn_probs

def minimal_layerwise_regression(
    model,
    run_id: str,
    nn_inputs,
    nn_beliefs,
    nn_probs,
    device: str = "cpu",
    run_layerwise: bool = False,
    kf: list = None, # list of tuples, each tuple contains two lists: the indices of the training set and the indices of the test set
    prefix_to_indices: dict = None
):
    """
    Minimal script to:
      1) Extract each layer's activations
      2) Run regression layer by layer + 'all_layers_combined'
    
    Args:
        model: The loaded model
        run_id: The run ID (used to determine model type)
        nn_inputs: Neural network inputs
        nn_beliefs: Neural network beliefs
        nn_probs: Neural network probabilities
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing regression results for each layer
    """
    # Extract activations from each layer
    print("Extracting layer activations...")
    act_extractor = ActivationExtractor(device=device)
    nn_acts = act_extractor.extract_activations(
        model,
        nn_inputs,
        get_nn_type(run_id),
        relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
    )

    if 'input' in nn_acts and 'rrxor' in run_id:
        print(nn_acts['input'])

    # Run regression on combined layers
    print("\nConcatenating all layer activations...")
    combined_acts = _combine_layer_activations(nn_acts)
    print(f"All-layers concatenated shape: {combined_acts.shape}")
    print(f'a single layer act shape: {nn_acts[list(nn_acts.keys())[1]].shape}')

    print(f"Found {len(nn_acts)} layer activations.")
    print(nn_acts.keys())
    print('first act shape', nn_acts[list(nn_acts.keys())[0]].shape)

    if 'input' in nn_acts and 'rrxor' in run_id:
        print(f'shape of input: {nn_acts["input"].shape}')

    # Setup for regression
    regression_analyzer = RegressionAnalyzer(device=device, use_efficient_pinv=True)
    sample_weights = nn_probs / nn_probs.sum()
    layer_results = {}
    
    # Run regression on individual layers
    print("\nRunning regression for each layer individually...")

    if run_layerwise:
        for layer_key, layer_act in nn_acts.items():
            
            deduplicated_activations, deduplicated_probs, deduplicated_beliefs, unique_prefixes = combine_duplicate_data(prefix_to_indices, layer_act, nn_probs, nn_beliefs, debug=False, tolerance=1e-6)
            if 'input' in layer_key and 'rrxor' in run_id:
                print(f'shape of deduplicated input: {deduplicated_activations.shape}')
                print(f'the activations are: {deduplicated_activations}')
                # check if sum of every row is 1
                print(f'sum of every row: {deduplicated_activations.sum(dim=1)}')
                print(f'is it true: {torch.allclose(deduplicated_activations.sum(dim=1), torch.ones_like(deduplicated_activations.sum(dim=1)))}')
            #layer_results[layer_key] = _run_regression_for_layer(
            #    regression_analyzer, layer_act, nn_beliefs, sample_weights
            #)
            layer_results[layer_key] = _run_regression_for_layer_flat(
                regression_analyzer, deduplicated_activations, deduplicated_beliefs, deduplicated_probs, kf
            )
            if 'input' in layer_key and 'rrxor' in run_id:
                print(f'shape of deduplicated input: {deduplicated_activations.shape}')
                print(f'the activations are: {deduplicated_activations}')
                # check if sum of every row is 1
                print(f'sum of every row: {deduplicated_activations.sum(dim=1)}')
                print(f'is it true: {torch.allclose(deduplicated_activations.sum(dim=1), torch.ones_like(deduplicated_activations.sum(dim=1)))}')
    


    deduplicated_activations, deduplicated_probs, deduplicated_beliefs, unique_prefixes = combine_duplicate_data(prefix_to_indices, combined_acts, nn_probs, nn_beliefs, debug=False, tolerance=1e-6)
    
    layer_results["all_layers_combined"] = _run_regression_for_layer_flat(
        regression_analyzer, deduplicated_activations, deduplicated_beliefs, deduplicated_probs, kf
    )
    
    print("\nDone! Returning results.")

    # if nn_type is transformer then convert the keys to layers
    if get_nn_type(run_id) == 'transformer':
        new_layer_results = {}
        for k, v in layer_results.items():
            if 'blocks.0.hook_resid_pre' in k:
                new_layer_results['input'] = v
            elif 'blocks.0.hook_resid_post' in k:
                new_layer_results['layer0'] = v
            elif 'blocks.1.hook_resid_post' in k:
                new_layer_results['layer1'] = v
            elif 'blocks.2.hook_resid_post' in k:
                new_layer_results['layer2'] = v
            elif 'blocks.3.hook_resid_post' in k:
                new_layer_results['layer3'] = v
            elif 'ln_final.hook_normalized' in k:
                new_layer_results['layer3_norm'] = v
            else:
                new_layer_results[k] = v
        layer_results = new_layer_results
        
    return layer_results


def _run_regression_for_layer(regression_analyzer, activations, beliefs, weights):
    """Helper function to run regression for a single layer or combined layers"""
    return run_single_rcond_sweep_with_predictions(
        regression_analyzer,
        activations,
        beliefs,
        weights,
        rcond_values=RCOND_SWEEP_LIST,
    )

def _run_regression_for_layer_flat(regression_analyzer, activations, beliefs, weights, kf):
    """Helper function to run regression for a single layer or combined layers"""

    return run_activation_to_beliefs_regression_kf(
        regression_analyzer,
        activations,
        beliefs,
        weights,
        kf,
        rcond_values=RCOND_SWEEP_LIST)


def _combine_layer_activations(nn_acts):
    """Combine activations from all layers into a single tensor, by concatenating them
    make sure to reshape back to the original shape
    a single act is of shape (n_samples, n_ctx, d_model)
    so in the end the shape should be (n_samples, n_ctx, d_model*n_layers)"""
    flattened_acts = []
    first_layer_key = list(nn_acts.keys())[0]
    n_samples = nn_acts[first_layer_key].shape[0]
    n_ctx = nn_acts[first_layer_key].shape[1]
    
    all_acts = []
    for layer_act in nn_acts.values():
        # Reshape to (n_samples, -1) to flatten any extra dimensions
        all_acts.append(layer_act)

    cat = torch.cat(all_acts, dim=2) 
    print(cat.shape)
    return cat

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from epsilon_transformers.visualization.plots import _project_to_simplex
from sklearn.decomposition import PCA

def visualize_activation_results(
    all_results, 
    layer='all_layers_combined',
    output_path=None,
    point_size={'truth': 0.15, 'pred': 0.1},
    transformation='cbrt',
    min_alpha=0.01,
    com = False,
    project_to_simplex=False,
    use_pca = False,
    inds_to_plot = [1,2],
    text_bottom_left = None,
    method = 'new_method',
):
    """
    Visualize activation analysis results across multiple checkpoints.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all checkpoints.
    layer : str, default='all_layers_combined'
        Which layer to use for visualization.
    output_path : str, optional
        Path to save the figure. If None, the figure is displayed instead.
    point_size : dict, default={'truth': 0.15, 'pred': 0.1}
        Size of scatter points for ground truth and predictions.
    transformation : str, default='cbrt'
        Method for weight transformation ('log', 'sqrt', 'cbrt').
    min_alpha : float, default=0.01
        Minimum alpha/transparency value.
    com : bool, default=False
        Whether to plot the center of masses for each of the unique belief states.
    project_to_simplex : bool, default=False
        Whether to project the data to the simplex.
    use_pca : bool, default=False
        Whether to use PCA of the true values to reduce the dimensionality of the data.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    # Function to transform weights for alpha values
    def transform_for_alpha(weights, min_alpha=0.1, transformation='log'):
        # Avoid zeros for log transformation
        epsilon = 1e-10
        
        if transformation == 'log':
            # Log transformation makes small values more visible
            transformed = np.log10(weights + epsilon) - np.log10(epsilon)
        elif transformation == 'sqrt':
            # Square root is less aggressive than log
            transformed = np.sqrt(weights)
        elif transformation == 'cbrt':
            # Cube root, between log and sqrt in aggressiveness
            transformed = np.cbrt(weights)
        else:
            transformed = weights
        
        # Scale to [0, 1] range
        if transformed.max() > transformed.min():
            normalized = (transformed - transformed.min()) / (transformed.max() - transformed.min())
        else:
            normalized = np.ones_like(transformed) * 0.5
        
        # Apply minimum alpha to ensure all points have some visibility
        return min_alpha + (1.0 - min_alpha) * normalized
    
    # Get the checkpoint names
    checkpoint_names = list(all_results.keys())
    print('CHECKPOINT NAMES', checkpoint_names)
    
    # Get first checkpoint results to extract common data
    results = all_results[checkpoint_names[0]]
    
    # Check if the specified layer exists
    if layer not in results:
        raise ValueError(f"Layer '{layer}' not found in results. Available layers: {list(results.keys())}")
    
    # Generate alpha values from weights
    weights = results[layer]['weights']
    alpha_values = transform_for_alpha(weights, min_alpha=min_alpha, transformation=transformation)
    



    # Create the scatter plot with multiple subplots side by side
    fig = plt.figure(figsize=(12.5, 2.5))
    
    # Get ground truth values
    x_true = results[layer]['true_values'][:, inds_to_plot[0]]
    y_true = results[layer]['true_values'][:, inds_to_plot[1]]
    if project_to_simplex:
        x_true, y_true = _project_to_simplex(results[layer]['true_values'])
    if use_pca:
        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(results[layer]['true_values'])
        x_true = pca_results[:, inds_to_plot[0]]
        y_true = pca_results[:, inds_to_plot[1]]
    
    # Calculate colors based on ground truth values
    R = (x_true - np.min(x_true)) / (np.max(x_true) - np.min(x_true)) if np.max(x_true) > np.min(x_true) else np.ones_like(x_true) * 0.5
    G = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true)) if np.max(y_true) > np.min(y_true) else np.ones_like(y_true) * 0.5
    B = np.sqrt(x_true**2 + y_true**2) / np.max(np.sqrt(x_true**2 + y_true**2))
    
    # Stack RGB and alpha for plotting
    colors_rgba = np.column_stack((R, G, B, alpha_values))
    
    # First subplot: Ground Truth
    ax1 = plt.subplot(1, len(checkpoint_names) + 1, 1)
    sc1 = ax1.scatter(x_true, y_true, color=colors_rgba, s=point_size['truth'])
    ax1.set_axis_off()
    ax1.set_title('Theoretical Prediction')
    
    # Get the x and y limits from the ground truth plot to apply to all subplots
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    
    # Create subplots for each checkpoint
    for i, checkpoint in enumerate(checkpoint_names):
        print('PLOTTING', checkpoint)
        
        checkpoint_results = all_results[checkpoint]
        
        # Get prediction values for this checkpoint
        x_pred = checkpoint_results[layer]['predictions'][:, inds_to_plot[0]]
        y_pred = checkpoint_results[layer]['predictions'][:, inds_to_plot[1]]
        if project_to_simplex:
            x_pred, y_pred = _project_to_simplex(checkpoint_results[layer]['predictions'])
        if use_pca:
            pred_results = pca.transform(checkpoint_results[layer]['predictions'])
            x_pred = pred_results[:, inds_to_plot[0]]
            y_pred = pred_results[:, inds_to_plot[1]]
        # Create subplot

        # Add these print statements in your visualize_activation_results function
        print(f"Truth data shape: {len(weights)}")  # Should match colors_rgba length
        print(f"x_true shape: {len(x_true)}")  # Should also match
        print(f"colors_rgba shape: {colors_rgba.shape}")  # This is 350 according to error

        # For each checkpoint processing:
        print(f"Checkpoint: {checkpoint}")
        print(f"Prediction shape: {checkpoint_results[layer]['predictions'].shape}")
        print(f"x_pred shape: {len(x_pred)}, y_pred shape: {len(y_pred)}")

        ax = plt.subplot(1, len(checkpoint_names) + 1, i+2)
        sc = ax.scatter(x_pred, y_pred, color=colors_rgba, s=point_size['pred'])

        # if com is true, plot the center of masses for each of the unique belief states
        if com:
            # compute the unique belief states, by looking at the beliefs
            beliefs_ = checkpoint_results[layer]['true_values']
            # find inds of each unique belief state
            unique_beliefs = np.unique(beliefs_, axis=0)
            for unique_belief in unique_beliefs:
                # find inds of each unique belief state
                unique_inds = np.where(np.all(beliefs_ == unique_belief, axis=1))[0]
                
                # compute the center of mass for each unique belief state
                com_x = np.mean(x_pred[unique_inds])
                com_y = np.mean(y_pred[unique_inds])
                # find the mean color of the points in the unique belief state
                com_color = np.mean(colors_rgba[unique_inds], axis=0)
                # except for the alpha value we should add across the unique inds
                com_color[3] = np.sum(colors_rgba[unique_inds, 3])
                # then put this through the transformation function
                com_color[3] = transform_for_alpha(com_color[3], min_alpha=min_alpha, transformation=transformation)
                ax.scatter(com_x, com_y, color=com_color, s=50, marker='o')

        ax.set_axis_off()
        
        # Apply the same x and y limits as the ground truth plot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Format checkpoint name for title
        checkpoint_title = checkpoint.replace('.pt', '')
        if checkpoint_title.isdigit() and int(checkpoint_title) == 0:
            checkpoint_title = "Initial"
        else:
            # Try to format large numbers more nicely
            try:
                checkpoint_num = int(checkpoint_title)
                if checkpoint_num >= 1e6:
                    checkpoint_title = f"{checkpoint_num/1e6:.1f}M"
            except:
                pass
                
        ax.set_title(checkpoint_title)
    
    # Create a color bar that shows alpha transformation
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    
    # Create colormap for transparency visualization
    weight_values = np.linspace(0, np.max(weights), 10000)
    alpha_for_cbar = transform_for_alpha(weight_values, min_alpha=min_alpha, transformation=transformation)
    
    # Create a custom colormap where color is constant but alpha varies
    rgba_colors = np.ones((len(alpha_for_cbar), 4))
    rgba_colors[:, 0] = 0  # R - black
    rgba_colors[:, 1] = 0  # G - black
    rgba_colors[:, 2] = 0  # B - black
    rgba_colors[:, 3] = alpha_for_cbar  # Alpha
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('alpha_cmap', rgba_colors)
    
    # Use LogNorm for better representation of weight distribution
    norm = LogNorm(vmin=max(np.min(weights), 1e-10), vmax=np.max(weights))
    
    # Create ScalarMappable with the custom colormap
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(weights)
    
    # Add colorbar with appropriate ticks
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Probability', rotation=270, labelpad=15)
    
    # Generate logarithmically spaced ticks
    max_order = int(np.floor(np.log10(np.max(weights))))
    min_order = int(np.floor(np.log10(max(np.min(weights), 1e-10))))
    ticks = [10**i for i in range(min_order, max_order+1)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])

    if text_bottom_left:
        # put text very small in the bottom left of the entire figure
        fig.text(0.01, 0.01, text_bottom_left, fontsize=8)
    
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig


#%%
# Define the process-to-run mapping (from your Google Doc table)
PROCESS_RUN_MAP = {
    "Mess3": {
        "RNN": ("20241121152808", "run_71"),
        "GRU": ("20241121152808", "run_63"),
        "LSTM": ("20241121152808", "run_55"),
        "Transformer": ("20241205175736", "run_23"),
    },
    "RRXor": {
        "RNN": ("20241121152808", "run_70"),
        "GRU": ("20241121152808", "run_62"),
        "LSTM": ("20241121152808", "run_54"),
        "Transformer": ("20241205175736", "run_22"),
    },
    "Fanizza": { # Assuming Fanizza is the FRDN process
        "RNN": ("20241121152808", "run_69"),
        "GRU": ("20241121152808", "run_61"),
        "LSTM": ("20241121152808", "run_53"),
        "Transformer": ("20241205175736", "run_21"),
    },
    "TomQuantumA": { # Assuming Tom Quantum A
        "RNN": ("20241121152808", "run_65"),
        "GRU": ("20241121152808", "run_57"),
        "LSTM": ("20241121152808", "run_49"),
        "Transformer": ("20241205175736", "run_17"),
    },
    "TomQuantumB": { # Assuming Tom Quantum B
        "RNN": ("20241121152808", "run_68"),
        "GRU": ("20241121152808", "run_60"),
        "LSTM": ("20241121152808", "run_52"),
        "Transformer": ("20241205175736", "run_20"),
    },
    "PostQuantum": { # Assuming Post Quantum
        "RNN": ("20241121152808", "run_64"),
        "GRU": ("20241121152808", "run_56"),
        "LSTM": ("20241121152808", "run_48"),
        "Transformer": ("20241205175736", "run_16"),
    },
}

OUTPUT_DIR = "./analysis_results" # Directory to save the results
DEVICE = "cpu" # Set device explicitly to CPU
N_SPLITS_KFOLD = 10 # Number of splits for K-Fold CV
RANDOM_STATE_KFOLD = 42
VARIANCE_THRESHOLD_FOR_METRIC = 0.99 # Threshold for calculating the single dim metric

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
# Clear previous handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%% [markdown]
# ## Helper Function: Dimensionality Calculation

#%%
def calculate_cumulative_variance(singular_values):
    """
    Calculates the cumulative variance explained by principal components.

    Args:
        singular_values (np.ndarray or torch.Tensor): Array of singular values from SVD.
            Should be sorted in descending order.

    Returns:
        np.ndarray: Cumulative variance explained by component (or empty array if input is None).
        np.ndarray: The processed singular values as a numpy array (or empty array if input is None).
    """
    if singular_values is None:
        return np.array([]), np.array([])

    # Ensure input is a numpy array on CPU
    if isinstance(singular_values, torch.Tensor):
        singular_values_np = singular_values.detach().cpu().numpy()
    elif isinstance(singular_values, (list, tuple)):
         singular_values_np = np.array(singular_values)
    elif isinstance(singular_values, np.ndarray):
        singular_values_np = singular_values
    else:
        # Let potential errors propagate
        raise TypeError(f"Unsupported type for singular_values: {type(singular_values)}")

    if singular_values_np.ndim != 1:
         singular_values_np = singular_values_np.flatten()

    if len(singular_values_np) == 0:
        return np.array([]), singular_values_np

    # Ensure non-negative values and handle potential NaNs
    singular_values_np = np.nan_to_num(singular_values_np, nan=0.0)
    singular_values_np = np.maximum(singular_values_np, 0.0)

    # Calculate variance explained
    total_variance = np.sum(singular_values_np**2)
    if total_variance < 1e-12:
        return np.zeros(len(singular_values_np)), singular_values_np

    variance_explained = (singular_values_np**2) / total_variance
    cumulative_variance = np.cumsum(variance_explained)
    cumulative_variance = np.minimum(cumulative_variance, 1.0) # Clip at 1.0

    return cumulative_variance, singular_values_np

#%% [markdown]
# ## Main Analysis Loop

#%%
# Initialize S3 loader
# Errors during initialization will stop the script here
s3_loader = S3ModelLoader(use_company_credentials=True)

for process_name, models in PROCESS_RUN_MAP.items():
    logging.info(f"--- Starting Analysis for Process: {process_name} ---")
    for model_type, (sweep_id, run_name) in models.items():
        logging.info(f"--- Analyzing Model Type: {model_type} (Sweep: {sweep_id}, Run: {run_name}) ---")
        
        runs = s3_loader.list_runs_in_sweep(sweep_id)
        # the run_id is the run that contains the run_name
        run_id = [x for x in runs if run_name in x][0]
        run_output_file = os.path.join(OUTPUT_DIR, f"{process_name}_{model_type}_{sweep_id}_{run_name}_analysis.pkl")

        # Skip if results already exist
        if os.path.exists(run_output_file):
            logging.warning(f"Results file already exists for {run_id}, skipping: {run_output_file}")
            continue

        if 'rrxor' not in run_id:
            continue

        # --- Data Loading and Preparation (Done Once Per Run) ---
        logging.info("Listing and selecting checkpoints...")
        checkpoints = s3_loader.list_checkpoints(sweep_id, run_id)
        if not checkpoints:
             # Let FileNotFoundError be raised if select_checkpoints expects a non-empty list
             # or handle inside select_checkpoints if preferred
             logging.warning(f"No checkpoints listed for sweep {sweep_id}, run {run_id}. Behavior depends on select_checkpoints.")
             # Depending on select_checkpoints, this might raise an error or return empty

        # Assume select_checkpoints handles empty list or raises error
        selected_checkpoints = select_checkpoints(s3_loader, sweep_id, run_id, checkpoints)
        if not selected_checkpoints:
             raise ValueError(f"No checkpoints were selected for run {run_id}.")
        logging.info(f"Selected {len(selected_checkpoints)} checkpoints: {[os.path.basename(c) for c in selected_checkpoints]}")

        logging.info("Loading initial data (inputs, beliefs, probs)...")
        # Assume prepare_msp_data_for_regression raises error on failure
        _, run_config, nn_inputs, nn_beliefs, nn_probs = prepare_msp_data_for_regression(
            sweep_id, run_id, selected_checkpoints[0], device=DEVICE
        )
        logging.info(f"Data loaded. Input shape: {nn_inputs.shape}, Beliefs shape: {nn_beliefs.shape}, Probs shape: {nn_probs.shape}")

        logging.info("Finding duplicate prefixes and preparing K-Fold splits...")
        prefix_to_indices = find_duplicate_prefixes(nn_inputs)


        # Assume deduplicate_tensor raises error on failure
        dedup_probs, _ = deduplicate_tensor(prefix_to_indices, nn_probs, aggregation_fn=sum)
        dedup_probs = dedup_probs.squeeze()

        # Assume compute_kfold_split raises error on failure
        kf, all_positions = compute_kfold_split(dedup_probs, n_splits=N_SPLITS_KFOLD, random_state=RANDOM_STATE_KFOLD)
        kf_list = list(kf.split(all_positions))
        logging.info(f"Prepared {len(kf_list)} K-Fold splits.")

        # --- Per-Checkpoint Analysis ---
        run_analysis_results = {} # Store results for all checkpoints of this run
        for checkpoint_path in tqdm(selected_checkpoints, desc=f"Analyzing {run_id}", leave=False):
            checkpoint_name = os.path.basename(checkpoint_path)
            logging.info(f"Processing checkpoint: {checkpoint_name}")

            # Load the specific model checkpoint - Assume raises error on failure
            model, _ = s3_loader.load_checkpoint(sweep_id, run_id, checkpoint_path, device=DEVICE)

            # Run the core regression analysis - Assume raises error on failure
            layer_results = minimal_layerwise_regression(
                model=model,
                run_id=run_id,
                nn_inputs=nn_inputs,
                nn_beliefs=nn_beliefs,
                nn_probs=nn_probs,
                device=DEVICE,
                run_layerwise=True,
                kf=kf_list,
                prefix_to_indices=prefix_to_indices
            )


            # --- Post-process results: Extract dimensionality ---
            for layer_name, results_dict in layer_results.items():
                singular_values_raw = results_dict.get('singular_values', None)

                # Initialize keys
                results_dict['singular_values'] = None
                results_dict['cumulative_variance_explained'] = None
                results_dict[f'dimensionality_{int(VARIANCE_THRESHOLD_FOR_METRIC*100)}_variance'] = None

                if singular_values_raw is not None:
                   cumulative_variance, singular_values_np = calculate_cumulative_variance(singular_values_raw)

                   if len(cumulative_variance) > 0:
                       num_components_threshold = np.searchsorted(cumulative_variance, VARIANCE_THRESHOLD_FOR_METRIC, side='right') + 1
                       results_dict['singular_values'] = singular_values_np
                       results_dict['cumulative_variance_explained'] = cumulative_variance
                       results_dict[f'dimensionality_{int(VARIANCE_THRESHOLD_FOR_METRIC*100)}_variance'] = num_components_threshold

            run_analysis_results[checkpoint_name] = layer_results
            logging.info(f"Finished processing checkpoint: {checkpoint_name}")

            # Clean up model explicitly after use in the loop
            del model
            if 'layer_results' in locals(): del layer_results


        # --- Save Results for the Entire Run ---
        if run_analysis_results:
            logging.info(f"Saving analysis results for run {run_id} to {run_output_file}")
            # Assume pickle.dump raises error on failure
            with open(run_output_file, 'wb') as f:
                pickle.dump(run_analysis_results, f)
        else:
            # This case might not be reachable if errors halt execution earlier
            logging.warning(f"No results generated for run {run_id}. Nothing to save.")

        logging.info(f"--- Finished Model Type: {model_type} ---")
    logging.info(f"--- Finished Process: {process_name} ---")


# No finally block, script ends or errors out before here

# %%
# now load the pkl files and make the figures

#%%
import pickle
process_name = 'RRXor'
model_type = 'RNN'
sweep_id = '20241121152808'
run_name = 'run_70'
run_output_file = os.path.join(OUTPUT_DIR, f"{process_name}_{model_type}_{sweep_id}_{run_name}_analysis.pkl")

# load the pkl file
with open(run_output_file, 'rb') as f:
    run_analysis_results = pickle.load(f)

#%%
print(run_analysis_results.keys())
preds = run_analysis_results['4075724800.pt']['input']['predictions']
truth = run_analysis_results['4075724800.pt']['input']['true_values']

#%%
# how many unique rows are there in truth?
print(len(np.unique(truth, axis=0)))

#%%
# how many unique rows are there in preds?
print(len(np.unique(preds, axis=0)))

# run PCA to make preds 2d and then scatter plot
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(truth)
truth_2d = pca.transform(truth)
preds_2d = pca.transform(preds)
#%%
plt.scatter(truth_2d[:, 0], truth_2d[:, 2], alpha=0.1, s=10)
plt.scatter(preds_2d[:, 0], preds_2d[:, 2], c='r', alpha=0.01, s=10)
plt.colorbar()
plt.show()


# %%
