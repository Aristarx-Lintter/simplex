import datashader as ds
import datashader.transfer_functions as tf
from datashader import reductions as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holoviews as hv
from sklearn.decomposition import PCA
import torch

def plot_predictions_datashader(results, figsize=(20, 10)):
    """
    Plot true values and predictions using datashader for aggregation.
    
    Args:
        results: Dictionary containing regression results with 'true_values' and 'predictions'
        figsize: Figure size tuple (width, height)
        
    Returns:
        Matplotlib figure with side-by-side datashader plots
    """
    # Extract data from results
    true_values = results['all_layers_combined']['true_values']
    predictions = results['all_layers_combined']['predictions']
    weights = results.get('all_layers_combined').get('weights', None)
    
    # If tensors, convert to numpy
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if weights is not None and isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    else:
        # If weights not available, use uniform weights
        weights = np.ones(len(true_values)) / len(true_values)
    
    # Determine dimensionality
    n_dims = true_values.shape[1]
    print(f"Data dimensionality: {n_dims}")
    
    # For dims > 3, use PCA to reduce to 3D for color mapping
    if n_dims > 3:
        pca = PCA(n_components=3)
        true_values_color = pca.fit_transform(true_values)
        print("Applied PCA to reduce dimensions to 3 for color mapping")
    else:
        true_values_color = true_values.copy()
    
    # Create color channels according to requirements:
    # R = x, G = y, Z = x^2+y^2 (assuming this is what was meant by x^2+r^2)
    r_channel = true_values_color[:, 0]  # R = x
    g_channel = true_values_color[:, 1]  # G = y
    b_channel = true_values_color[:, 0]**2 + true_values_color[:, 1]**2  # Z = x^2+y^2
    
    # Normalize color channels to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val != max_val:
            return (arr - min_val) / (max_val - min_val)
        return np.ones_like(arr) * 0.5
    
    r_norm = normalize(r_channel)
    g_norm = normalize(g_channel)
    b_norm = normalize(b_channel)
    
    # Create DataFrames for datashader
    df_true = pd.DataFrame({
        'x': true_values[:, 0],
        'y': true_values[:, 1],
        'alpha': weights,
        'r': r_norm,
        'g': g_norm,
        'b': b_norm
    })
    
    df_pred = pd.DataFrame({
        'x': predictions[:, 0],
        'y': predictions[:, 1],
        'alpha': weights,
        'r': r_norm,  # Using colors from true values
        'g': g_norm,
        'b': b_norm
    })
    
    # Determine plot ranges to make both plots use the same scale
    x_min = min(df_true['x'].min(), df_pred['x'].min())
    x_max = max(df_true['x'].max(), df_pred['x'].max())
    y_min = min(df_true['y'].min(), df_pred['y'].min())
    y_max = max(df_true['y'].max(), df_pred['y'].max())
    
    # Add some padding
    x_range = (x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    y_range = (y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Helper function for datashader rendering
    def render_datashader(df, ax, title):
        # Create canvas with the same dimensions and ranges for both plots
        canvas = ds.Canvas(
            plot_width=800, 
            plot_height=800,
            x_range=x_range,
            y_range=y_range
        )
        
        # Aggregate points, weighted by alpha
        agg = canvas.points(df, 'x', 'y', rd.sum('alpha'))
        
        # Compute color aggregations
        r_agg = canvas.points(df, 'x', 'y', rd.sum('r'))
        g_agg = canvas.points(df, 'x', 'y', rd.sum('g'))
        b_agg = canvas.points(df, 'x', 'y', rd.sum('b'))
        
        # Normalize by the weight aggregation
        r_img = tf.shade(r_agg, cmap=['black', 'red'], how='linear')
        g_img = tf.shade(g_agg, cmap=['black', 'green'], how='linear')
        b_img = tf.shade(b_agg, cmap=['black', 'blue'], how='linear')
        
        # Combine RGB channels
        img = tf.stack(r_img, g_img, b_img)
        
        # Apply alpha from point density
        alpha_img = tf.shade(agg, cmap=['transparent', 'white'], how='log')
        final_img = tf.combine_alpha(img, alpha_img)
        
        # Display
        ax.imshow(np.array(final_img), extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
    # Render both plots
    render_datashader(df_true, axes[0], "True Values")
    render_datashader(df_pred, axes[1], "Predictions")
    
    plt.tight_layout()
    return fig

def plot_predictions_holoviews(results):
    """
    Alternative implementation using HoloViews for interactive datashader plots.
    
    Args:
        results: Dictionary containing regression results
    
    Returns:
        HoloViews layout with interactive datashader plots
    """
    # Initialize holoviews
    hv.extension('bokeh')
    
    # Extract data from results
    true_values = results['all_layers_combined']['true_values']
    predictions = results['all_layers_combined']['predictions']
    
    # If tensors, convert to numpy
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Get weights (nn_probs) for alpha values
    weights = results.get('all_layers_combined').get('weights', None)
    if weights is not None and isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    else:
        weights = np.ones(len(true_values)) / len(true_values)
    
    # Determine dimensionality
    n_dims = true_values.shape[1]
    
    # For dims > 3, use PCA to reduce to 3D for color mapping
    if n_dims > 3:
        pca = PCA(n_components=3)
        true_values_color = pca.fit_transform(true_values)
    else:
        true_values_color = true_values.copy()
    
    # Create color channels
    r_channel = true_values_color[:, 0]  # R = x
    g_channel = true_values_color[:, 1]  # G = y
    b_channel = true_values_color[:, 0]**2 + true_values_color[:, 1]**2  # Z = x^2+y^2
    
    # Normalize color channels to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val != max_val:
            return (arr - min_val) / (max_val - min_val)
        return np.ones_like(arr) * 0.5
    
    r_norm = normalize(r_channel)
    g_norm = normalize(g_channel)
    b_norm = normalize(b_channel)
    
    # Create DataFrames
    df_true = pd.DataFrame({
        'x': true_values[:, 0],
        'y': true_values[:, 1],
        'alpha': weights,
        'r': r_norm,
        'g': g_norm,
        'b': b_norm
    })
    
    df_pred = pd.DataFrame({
        'x': predictions[:, 0],
        'y': predictions[:, 1],
        'alpha': weights,
        'r': r_norm,
        'g': g_norm,
        'b': b_norm
    })
    
    # Create scatter points with datashader
    true_points = hv.Points(df_true, kdims=['x', 'y'], vdims=['alpha', 'r', 'g', 'b'])
    pred_points = hv.Points(df_pred, kdims=['x', 'y'], vdims=['alpha', 'r', 'g', 'b'])
    
    # Apply datashader to both plots
    datashade = hv.operation.datashader.datashade
    
    # Create colored datashaded points
    true_dshaded = datashade(true_points, aggregator=rd.sum('alpha'))
    pred_dshaded = datashade(pred_points, aggregator=rd.sum('alpha'))
    
    # Create layout
    layout = hv.Layout([
        true_dshaded.relabel('True Values'),
        pred_dshaded.relabel('Predictions')
    ]).cols(2)
    
    return layout

# Example usage:
# fig = plot_predictions_datashader(results)
# plt.show()
# 
# # Or for interactive plotting:
# plot = plot_predictions_holoviews(results)
# plot 