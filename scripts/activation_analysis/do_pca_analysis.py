# %%
# add the autoreload
%load_ext autoreload
%autoreload 2

import torch
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.regression import (
    RegressionAnalyzer,
    run_single_rcond_sweep_with_predictions,
    run_single_rcond_sweep_with_predictions_flat,
    run_paul_rcond_sweep_with_sklearn_predictions_flat,
    run_activation_to_beliefs_regression_kf,
    run_activation_to_beliefs_regression_ridgecv,
    compute_weighted_pca_variance
)
from scripts.activation_analysis.config import RCOND_SWEEP_LIST
from typing import Optional, List, Dict, Tuple
import numpy as np # Make sure numpy is imported
import os # Make sure os is imported


# %%


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

def compute_layerwise_pca_variance(
    model,
    run_id: str,
    nn_inputs,
    nn_beliefs,
    nn_probs,
    device: str = "cpu",
    run_layerwise: bool = False,
    prefix_to_indices: Optional[Dict] = None
):
    """
    Computes weighted PCA variance for each layer's activations.
    
    Args:
        model: The loaded model
        run_id: The run ID (used to determine model type)
        nn_inputs: Neural network inputs
        nn_beliefs: Neural network beliefs
        nn_probs: Neural network probabilities
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing PCA results for each layer
    """
    # Extract activations from each layer
    act_extractor = ActivationExtractor(device=device)
    nn_acts = act_extractor.extract_activations(
        model,
        nn_inputs,
        get_nn_type(run_id),
        relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
    )
    
    # Initialize results dictionary
    layer_results = {}
    
    # Run PCA variance analysis on individual layers
    if run_layerwise:
        print("\nRunning PCA variance analysis for each layer...")
        for layer_key, layer_act in nn_acts.items():
            
            deduplicated_activations, deduplicated_probs, deduplicated_beliefs, unique_prefixes = combine_duplicate_data(prefix_to_indices, layer_act, nn_probs, nn_beliefs, debug=False, tolerance=1e-6)
            
            # Compute PCA variance
            B = deduplicated_beliefs.shape[-1]
            pca_cumulative_variance, pca_variance_at_B, pca_variance_at_B_minus_1 = compute_weighted_pca_variance(
                deduplicated_activations, deduplicated_probs, B
            )
            
            # Store only PCA results
            layer_results[layer_key] = {
                'pca_cumulative_variance': pca_cumulative_variance,
                'pca_variance_at_B': pca_variance_at_B,
                'pca_variance_at_B_minus_1': pca_variance_at_B_minus_1
            }
    
    # Run PCA variance analysis on combined layers
    print("\nRunning PCA variance analysis for combined layers...")
    deduplicated_activations, deduplicated_probs, deduplicated_beliefs, unique_prefixes = combine_duplicate_data(prefix_to_indices, _combine_layer_activations(nn_acts), nn_probs, nn_beliefs, debug=False, tolerance=1e-6)
    
    # Compute PCA variance calculation for combined layers
    B_combined = deduplicated_beliefs.shape[-1]
    pca_cumulative_variance_combined, pca_variance_at_B_combined, pca_variance_at_B_minus_1_combined = compute_weighted_pca_variance(
        deduplicated_activations, deduplicated_probs, B_combined
    )
    
    # Store only PCA results for combined layers
    layer_results["all_layers_combined"] = {
        'pca_cumulative_variance': pca_cumulative_variance_combined,
        'pca_variance_at_B': pca_variance_at_B_combined,
        'pca_variance_at_B_minus_1': pca_variance_at_B_minus_1_combined
    }
    
    print("\nPCA Variance computation done! Returning results.")

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


def plot_pca_variance_curves(results, checkpoint_name, layer_name, output_path):
    """
    Plots the cumulative explained variance curve from PCA results.

    Args:
        results (dict): The results dictionary containing PCA info.
        checkpoint_name (str): The specific checkpoint to plot.
        layer_name (str): The specific layer whose activations were analyzed.
        output_path (str): Path to save the plot image.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        layer_results = results[checkpoint_name][layer_name]
        cumulative_variance = layer_results.get('pca_cumulative_variance')
        variance_at_B = layer_results.get('pca_variance_at_B')
        variance_at_B_minus_1 = layer_results.get('pca_variance_at_B_minus_1')
        
        # Try to infer B from the shape of true_values if available
        B = None
        if 'true_values' in layer_results and layer_results['true_values'] is not None:
            B = layer_results['true_values'].shape[-1]
        elif 'beliefs' in layer_results and layer_results['beliefs'] is not None: # Fallback
             B = layer_results['beliefs'].shape[-1]
        print(f"B: {B}")

        if cumulative_variance is None:
            print(f"PCA variance data not found for {checkpoint_name} / {layer_name}. Skipping plot.")
            return
        
        if np.isnan(cumulative_variance).all():
             print(f"PCA variance data is all NaN for {checkpoint_name} / {layer_name} (SVD likely failed). Skipping plot.")
             return

        num_components = len(cumulative_variance)
        components = np.arange(1, num_components + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(components, cumulative_variance, marker='.', linestyle='-', label='Cumulative Variance')

        # Annotate B and B-1 if B is known and variance is valid
        if B is not None:
            if not np.isnan(variance_at_B):
                plt.plot(B, variance_at_B, 'ro', label=f'B={B} ({variance_at_B:.3f})')
                plt.vlines(B, 0, variance_at_B, colors='r', linestyles='dashed', alpha=0.7)
                plt.hlines(variance_at_B, 0, B, colors='r', linestyles='dashed', alpha=0.7)
            if not np.isnan(variance_at_B_minus_1):
                plt.plot(B - 1, variance_at_B_minus_1, 'go', label=f'B-1={B-1} ({variance_at_B_minus_1:.3f})')
                plt.vlines(B - 1, 0, variance_at_B_minus_1, colors='g', linestyles='dashed', alpha=0.7)
                plt.hlines(variance_at_B_minus_1, 0, B - 1, colors='g', linestyles='dashed', alpha=0.7)

        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title(f'Weighted PCA Explained Variance\nCheckpoint: {checkpoint_name}, Layer: {layer_name}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(0, 1.05)
        plt.xlim(0, num_components + 1)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=150)
        print(f"Saved PCA variance plot to {output_path}")

    except KeyError:
        print(f"Could not find results for checkpoint '{checkpoint_name}' or layer '{layer_name}'.")
    except Exception as e:
        print(f"An error occurred during PCA plotting for {checkpoint_name}/{layer_name}: {e}")


# %%
def select_checkpoints(s3_loader, sweep_id, run_id, checkpoints):
    """
    Select a subset of checkpoints for analysis based on validation loss or evenly spaced intervals.
    
    Args:
        s3_loader: The loader to access data from S3
        sweep_id: ID of the sweep
        run_id: ID of the run
        checkpoints: List of all available checkpoints
        
    Returns:
        List of selected checkpoint paths
    """
    # Attempt to load validation  loss data
    use_loss_spacing = False
    try:
        loss_df = s3_loader.load_loss_from_run(sweep_id, run_id)
        if loss_df is None or 'num_tokens_seen' not in loss_df.columns or 'val_loss_mean' not in loss_df.columns:
            print("Warning: Loss data not available or invalid. Using evenly spaced checkpoints.")
            use_loss_spacing = False
        else:
            print("Successfully loaded validation loss data.")
            use_loss_spacing = True
            val_loss_data = loss_df[['num_tokens_seen', 'val_loss_mean']]
            
            # Extract step numbers from checkpoint filenames
            checkpoint_steps = []
            step_to_idx = {}
            for i, ckpt in enumerate(checkpoints):
                filename = os.path.basename(ckpt)
                step = int(filename.replace(".pt", ""))
                checkpoint_steps.append(step)
                step_to_idx[step] = i
            
            # Map steps to loss values
            step_to_loss = {}
            for step in checkpoint_steps:
                closest_idx = (val_loss_data['num_tokens_seen'] - step).abs().idxmin()
                step_to_loss[step] = val_loss_data.loc[closest_idx, 'val_loss_mean']
    except Exception as e:
        print(f"Error loading loss data: {e}")
        print("Falling back to evenly spaced checkpoints.")
        use_loss_spacing = False

    # Select 4 specific checkpoints to analyze
    if len(checkpoints) >= 4:
        # Always include the first checkpoint (index 0)
        selected_indices = [0]
        
        # Add the first checkpoint after the initial one (index 1)
        selected_indices.append(1)
        
        # Always include the last checkpoint
        selected_indices.append(len(checkpoints) - 1)
        
        # Add a checkpoint based on validation loss
        if use_loss_spacing:
            try:
                # Sort checkpoints by loss
                sorted_by_loss = sorted(checkpoint_steps, key=lambda s: step_to_loss[s])
                min_loss_step = sorted_by_loss[-1]  # Lowest loss (usually last)
                max_loss_step = sorted_by_loss[0]   # Highest loss (usually first)
                
                # Calculate target loss value that's 0.3 from min_loss toward max_loss
                min_loss = step_to_loss[min_loss_step]
                max_loss = step_to_loss[max_loss_step]
                target_loss = min_loss + 0.3 * (max_loss - min_loss)
                
                # Find checkpoint with loss closest to target, excluding already selected checkpoints
                already_selected_steps = [checkpoint_steps[i] for i in selected_indices]
                available_steps = [s for s in checkpoint_steps if s not in already_selected_steps]
                
                if available_steps:
                    closest_step = min(available_steps, key=lambda s: abs(step_to_loss[s] - target_loss))
                    selected_indices.append(step_to_idx[closest_step])
                else:
                    # Fallback if no steps meet the criteria
                    print("Warning: Could not find a suitable checkpoint based on loss. Using middle checkpoint.")
                    middle_idx = len(checkpoints) // 2
                    if middle_idx not in selected_indices:
                        selected_indices.append(middle_idx)
            except Exception as e:
                print(f"Error during loss-based selection: {e}")
                print("Falling back to middle checkpoint.")
                middle_idx = len(checkpoints) // 2
                if middle_idx not in selected_indices:
                    selected_indices.append(middle_idx)
        else:
            # If not using loss spacing, use a checkpoint in the middle
            middle_idx = len(checkpoints) // 2
            if middle_idx not in selected_indices:
                selected_indices.append(middle_idx)
        
        selected_indices = sorted(list(set(selected_indices)))
        selected_checkpoints = [checkpoints[i] for i in selected_indices]
    else:
        # If we have fewer than 4 checkpoints, use all available
        selected_checkpoints = checkpoints

    if 'val_loss_data' in locals() and len(val_loss_data) > 0:
        print(val_loss_data.head())
        
    print(f"Selected {len(selected_checkpoints)} checkpoints for analysis")
    return selected_checkpoints

# Replace the original code block with a function call


# %%
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_checkpoints(sweep_id, run_id, s3_loader):
    """
    Analyze checkpoints from a specific run, running regression analysis on each.
    
    Parameters:
    -----------
    sweep_id : str
        The sweep ID to analyze
    run_id : str
        The run ID within the sweep to analyze
    s3_loader : object
        Loader object used to access data and checkpoints
        
    Returns:"
    --------
    dict
        Dictionary containing results of regression analysis for each checkpoint
    """
    # Load validation loss data and select checkpoints for analysis
    checkpoints = s3_loader.list_checkpoints(sweep_id, run_id)
    print(f"Found {len(checkpoints)} checkpoints")
    
    selected_checkpoints = select_checkpoints(s3_loader, sweep_id, run_id, checkpoints)
        
    print(f"Selected {len(selected_checkpoints)} checkpoints for analysis")

    # Initialize results dictionary
    all_results = {}

            # We need to load the model for each checkpoint!
    model, run_config, nn_inputs, nn_beliefs, nn_probs = prepare_msp_data_for_regression(
        sweep_id, run_id, selected_checkpoints[0], device="cpu"
    )
    
    # Run PCA variance analysis for all checkpoints using the same input data
    for i, checkpoint_path in enumerate(selected_checkpoints):
        print(f"Running PCA Variance Analysis for checkpoint {i+1}/{len(selected_checkpoints)}")

        # load the model for each checkpoint
        model, run_config = s3_loader.load_checkpoint(
            sweep_id, run_id, checkpoint_path, device="cpu"
        )

        prefix_to_indices = find_duplicate_prefixes(nn_inputs)
        dedup_probs, dedup_indices = deduplicate_tensor(prefix_to_indices, nn_probs, aggregation_fn=sum)

        results = compute_layerwise_pca_variance(
            model, run_id, nn_inputs, nn_beliefs, nn_probs, device="cpu", run_layerwise=True,
            prefix_to_indices=prefix_to_indices
        )
        
        all_results[checkpoint_path.split('/')[-1]] = results
        
    return all_results 

# %%
# Example usage TOM QUANTUM
# go into first key then the keys of that are the layer names


s3_loader = S3ModelLoader(use_company_credentials=True)
sweeps = s3_loader.list_sweeps()
#sweeps = ['20250327115247']
for sweep in sweeps:
    sweep_id = sweep
    runs = s3_loader.list_runs_in_sweep(sweep)
    for run in runs:
        run_id = run
        # if L4 not in run, skip
        print(run)
        if 'L4' not in run:
            print('skipping')
            continue
        
        checkpoints_results = analyze_checkpoints(sweep, run, s3_loader)
        print('CHECKPOINTS RESULTS', checkpoints_results.keys())
        first_key = list(checkpoints_results.keys())[0]
        print(checkpoints_results[first_key].keys())

        # --- Add PCA Plotting Call --- START ---
        # Get the name of the last checkpoint analyzed

        var_at_B_over_ckpt = []
        var_at_B_minus_1_over_ckpt = []
        ckpt_names = []
        for ckpt in checkpoints_results.keys():
            var_at_B_over_ckpt.append(checkpoints_results[ckpt]['all_layers_combined']['pca_variance_at_B'])
            var_at_B_minus_1_over_ckpt.append(checkpoints_results[ckpt]['all_layers_combined']['pca_variance_at_B_minus_1'])
            ckpt_names.append(ckpt)
        plt.plot(ckpt_names, var_at_B_over_ckpt, label='B')
        plt.plot(ckpt_names, var_at_B_minus_1_over_ckpt, label='B-1')
        # add horizontal line at 1 and 0.9 and .95
        plt.axhline(1, color='k', linestyle='--')
        plt.axhline(0.9, color='k', linestyle='--')
        plt.axhline(0.95, color='k', linestyle='--')
        plt.legend()
        plt.show()
        # save it
        plt.savefig(f'./pca_analysis_figs/{sweep_id}_{run_id}.png')

        # now make plot for layer3
        var_at_B_over_ckpt = []
        var_at_B_minus_1_over_ckpt = []
        ckpt_names = []
        for ckpt in checkpoints_results.keys():
            var_at_B_over_ckpt.append(checkpoints_results[ckpt]['layer3']['pca_variance_at_B'])
            var_at_B_minus_1_over_ckpt.append(checkpoints_results[ckpt]['layer3']['pca_variance_at_B_minus_1'])
            ckpt_names.append(ckpt)
        plt.plot(ckpt_names, var_at_B_over_ckpt, label='B')
        plt.plot(ckpt_names, var_at_B_minus_1_over_ckpt, label='B-1')
        plt.legend()
        plt.show()
        plt.savefig(f'./pca_analysis_figs/{sweep_id}_{run_id}_layer3.png')


# %%


# %%



