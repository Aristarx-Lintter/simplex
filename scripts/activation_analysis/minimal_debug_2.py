#%%
# Modular script for displaying belief states and predictions from multiple checkpoints

import os
import torch
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
import time  # Add time module for timing

# Import the required epsilon_transformers modules
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from epsilon_transformers.visualization.plots import _project_to_simplex
from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.regression import RegressionAnalyzer, run_single_rcond_sweep_with_predictions
from scripts.activation_analysis.config import RCOND_SWEEP_LIST

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Safe tensor to numpy conversion helper
def safe_to_numpy(tensor):
    """Convert a PyTorch tensor to NumPy array safely.
    
    Args:
        tensor: PyTorch tensor or NumPy array or other object
        
    Returns:
        NumPy array, or the original object if conversion not possible
    """
    if isinstance(tensor, np.ndarray):
        # Already a NumPy array
        return tensor
    elif hasattr(tensor, 'detach') and callable(tensor.detach):
        # It's a PyTorch tensor
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, 'numpy') and callable(tensor.numpy):
        # It might be a TensorFlow tensor or something with a numpy method
        return tensor.numpy()
    else:
        # Return as is (could be a list, scalar, etc.)
        return tensor

def create_consolidated_visualization_datashader(
    nn_beliefs, nn_probs,
    layer_predictions_dict,
    checkpoint_names,
    sweep_id=None,
    run_id=None,
    vis_params=None,
    layer_order=None
):
    """
    Datashader-based version of create_consolidated_visualization(...) 
    for 'tom_quantum' runs. 
    It does nearly the same row/column subplot arrangement, but instead
    of adding massive Scattergl traces, we rasterize via Datashader. 

    Args:
        nn_beliefs: Ground truth belief states (tensor)
        nn_probs: Ground truth probabilities (tensor)
        layer_predictions_dict: dict mapping layer names -> list of predictions
        checkpoint_names: list of checkpoint display names
        sweep_id, run_id: optional strings
        vis_params: dict of visualization params, e.g. 
            {
                "gt_size": 2,
                "pred_size": 1.5,
                "project_to_simplex": False,
                "inds": [0, 1],
                "coms": False,
                "use_pca": False,
                "pca_dims": 2,
                "min_alpha": 0.15
            }
        layer_order: optional list specifying which layers (and in what order)
    """
    if vis_params is None:
        vis_params = {
            "gt_size": 2,
            "pred_size": 1.5,
            "project_to_simplex": False,
            "inds": [0,1],
            "coms": False,
            "use_pca": False,
            "pca_dims": 2,
            "min_alpha": 0.15,
        }
    
    from epsilon_transformers.visualization.plots import _project_to_simplex
    
    # We'll reuse these small helper functions from your code if they exist
    # or define them inline if needed.
    base_dim = nn_beliefs.shape[-1]
    nn_beliefs_np = nn_beliefs.reshape(-1, base_dim)
    nn_probs_np = nn_probs.reshape(-1)

    # Decide how to compute the (R,G,B,alpha) per point
    if base_dim == 3:
        # Reuse your prepare_visualization_data
        # Ensure nn_beliefs and nn_probs are converted to numpy first
        nn_beliefs_np_safe = safe_to_numpy(nn_beliefs_np)
        nn_probs_np_safe = safe_to_numpy(nn_probs_np)
        _, _, color_strings, (R_int, G_int, B_int) = prepare_visualization_data(nn_beliefs_np_safe, nn_probs_np_safe, min_alpha=vis_params["min_alpha"])
        # parse out alpha
        alpha_vals = []
        for c in color_strings:
            # c is like 'rgba(r,g,b,a)'
            # we can extract the last piece
            a_str = c.split(',')[-1].replace(')','').strip()
            a_val = float(a_str)
            alpha_vals.append(a_val)
        alpha_vals = np.array(alpha_vals)
    else:
        # If you have >3 dims, do the fallback PCA approach for color
        from sklearn.decomposition import PCA
        nn_beliefs_np_safe = safe_to_numpy(nn_beliefs_np)  # Ensure it's numpy
        pcs_3 = PCA(n_components=3).fit_transform(nn_beliefs_np_safe)
        min_vals = pcs_3.min(axis=0)
        max_vals = pcs_3.max(axis=0)
        rng_vals = (max_vals - min_vals) + 1e-8

        nn_probs_np_safe = safe_to_numpy(nn_probs_np)  # Ensure it's numpy
        alpha_vals = compute_alpha_float(nn_probs_np_safe, scale=1, min_alpha=vis_params["min_alpha"])
        R_int, G_int, B_int = [], [], []
        for i in range(len(pcs_3)):
            rr = 255 * (pcs_3[i,0] - min_vals[0]) / rng_vals[0]
            gg = 255 * (pcs_3[i,1] - min_vals[1]) / rng_vals[1]
            bb = 255 * (pcs_3[i,2] - min_vals[2]) / rng_vals[2]
            R_int.append(rr)
            G_int.append(gg)
            B_int.append(bb)
        R_int = np.array(R_int)
        G_int = np.array(G_int)
        B_int = np.array(B_int)
    
    # Weighted PCA for the actual 2D projection, if requested
    use_pca = vis_params.get("use_pca", False)
    pca_dims = vis_params.get("pca_dims", 2)
    inds = vis_params.get("inds", [0,1])
    project_to_simplex = vis_params.get("project_to_simplex", False)
    coms = vis_params.get("coms", False)

    if use_pca:
        nn_beliefs_np_safe = safe_to_numpy(nn_beliefs_np)
        nn_probs_np_safe = safe_to_numpy(nn_probs_np)
        mu, comps = weighted_pca(nn_beliefs_np_safe, nn_probs_np_safe, n_components=pca_dims)
        nn_beliefs_proj = apply_pca_projection(nn_beliefs_np_safe, mu, comps)
        if pca_dims > 2 and inds is not None:
            nn_beliefs_2d = nn_beliefs_proj[:, inds]
        else:
            nn_beliefs_2d = nn_beliefs_proj[:, :2]
    else:
        if project_to_simplex and base_dim == 3:
            # Make sure we're working with tensors for _project_to_simplex
            if not isinstance(nn_beliefs, torch.Tensor):
                nn_beliefs_tensor = torch.tensor(nn_beliefs)
            else:
                nn_beliefs_tensor = nn_beliefs
            nn_beliefs_2d = safe_to_numpy(_project_to_simplex(nn_beliefs_tensor.reshape(-1, 3)))
        elif inds is not None:
            nn_beliefs_2d = safe_to_numpy(nn_beliefs_np[:, inds])
        else:
            nn_beliefs_2d = safe_to_numpy(nn_beliefs_np[:, :2])

    # Filter out layers that have no predictions
    valid_layers = {
        layer: preds for layer, preds in layer_predictions_dict.items() if preds
    }

    # If layer_order is set, we respect it
    if layer_order:
        layers_to_plot = [ly for ly in layer_order if ly in valid_layers]
    else:
        # fallback
        normal_layers = [ly for ly in valid_layers.keys() if ly != "all_layers_combined"]
        normal_layers.sort()
        layers_to_plot = normal_layers
        if "all_layers_combined" in valid_layers:
            layers_to_plot.append("all_layers_combined")

    # Determine # of checkpoints in each layer
    max_checkpoints = max((len(preds) for preds in valid_layers.values()), default=0)
    n_cols = max_checkpoints + 1
    n_rows = len(layers_to_plot)

    # Make subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=None,
        vertical_spacing=0.05,
        horizontal_spacing=0.01
    )

    fig_height = n_rows * 175
    fig_width  = n_cols * 150
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=80, r=20, t=30, b=30),
        height=fig_height,
        width=fig_width
    )

    # We'll define a helper for Datashader
    def datashader_rasterize(xvals, yvals, R, G, B, A, 
                             plot_width=200, plot_height=200):
        """
        Sums up R*A, G*A, B*A, and A in each pixel, returning an RGBA image array
        and the bounding box. 
        """
        # Ensure all inputs are converted to numpy arrays
        xvals_np = safe_to_numpy(xvals)
        yvals_np = safe_to_numpy(yvals)
        R_np = safe_to_numpy(R)
        G_np = safe_to_numpy(G)
        B_np = safe_to_numpy(B)
        A_np = safe_to_numpy(A)
        
        df = pd.DataFrame({
            'x': xvals_np,
            'y': yvals_np,
            'RA': R_np * A_np,
            'GA': G_np * A_np,
            'BA': B_np * A_np,
            'A': A_np
        })
        x_min, x_max = np.min(xvals_np), np.max(xvals_np)
        y_min, y_max = np.min(yvals_np), np.max(yvals_np)
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        
        cvs = ds.Canvas(
            plot_width=plot_width,
            plot_height=plot_height,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max)
        )
        agg_R = cvs.points(df, 'x', 'y', ds.sum('RA'))
        agg_G = cvs.points(df, 'x', 'y', ds.sum('GA'))
        agg_B = cvs.points(df, 'x', 'y', ds.sum('BA'))
        agg_A = cvs.points(df, 'x', 'y', ds.sum('A'))

        r_vals = agg_R.values
        g_vals = agg_G.values
        b_vals = agg_B.values
        a_vals = agg_A.values

        with np.errstate(divide='ignore', invalid='ignore'):
            r_final = r_vals / a_vals
            g_final = g_vals / a_vals
            b_final = b_vals / a_vals

        r_final[np.isnan(r_final)] = 0
        g_final[np.isnan(g_final)] = 0
        b_final[np.isnan(b_final)] = 0

        # We'll just make alpha=1 wherever a_vals>0 
        a_final = (a_vals>0).astype(np.float32)

        height, width = r_vals.shape
        # Create a white background RGBA image
        rgba_img = np.ones((height, width, 4), dtype=np.uint8) * 255
        # Only update RGB values where we actually have data
        mask = (a_vals > 0)
        rgba_img[mask, 0] = r_final[mask].astype(np.uint8)
        rgba_img[mask, 1] = g_final[mask].astype(np.uint8)
        rgba_img[mask, 2] = b_final[mask].astype(np.uint8)
        rgba_img[..., 3] = 255  # Set full opacity everywhere for white background

        return rgba_img, (x_min, x_max, y_min, y_max)

    # If coms=True, we do small scatter for COM points. 
    # Otherwise, we do big Datashader images in each cell.
    # We'll handle each row (layer) in turn
    row_idx = 1
    for layer_name in layers_to_plot:
        predictions_list = valid_layers[layer_name]  # list of predictions for each checkpoint

        # If no predictions, skip
        if not predictions_list:
            row_idx += 1
            continue

        # We'll also figure out how many columns (the last columns might be empty if 
        # some layer has fewer checkpoints). But typically you have the same # of checkpoints.
        # 1 col for ground truth + len(predictions_list)
        # col 1 => ground truth, col 2.. => checkpoints
        if coms:
            # ------------------
            # We'll do standard COM scatter, because it's presumably not large data
            # ------------------
            # 1) Group
            unique_dict = defaultdict(list)
            for i,b in enumerate(nn_beliefs_np_safe):
                key = tuple(np.round(b,5))
                unique_dict[key].append(i)
            
            # We'll want color arrays for each index
            color_array = np.array([ 
                (R_int[i], G_int[i], B_int[i], alpha_vals[i]) for i in range(len(nn_beliefs_np_safe))
            ])

            # 2) Ground truth COM
            com_ground_x = []
            com_ground_y = []
            com_colors   = []

            for b_key, idxs in unique_dict.items():
                group_w = nn_probs_np_safe[idxs] # Use the safe numpy version
                
                # Handle different types for nn_beliefs_2d
                if isinstance(nn_beliefs_2d, tuple) and len(nn_beliefs_2d) == 2:
                    # Direct tuple from _project_to_simplex - indices are already separated
                    x_vals = nn_beliefs_2d[0][idxs]
                    y_vals = nn_beliefs_2d[1][idxs]
                    
                    wsum = group_w.sum()
                    if wsum > 0:
                        com_x = np.sum(x_vals * group_w) / wsum
                        com_y = np.sum(y_vals * group_w) / wsum
                    else:
                        com_x = np.mean(x_vals)
                        com_y = np.mean(y_vals)
                else:
                    # Normal array case - need to handle different shapes
                    coords_2d = safe_to_numpy(nn_beliefs_2d[idxs]) # Ensure coords are numpy
                    
                    wsum = group_w.sum()
                    if wsum>0:
                        # Check dimensions of coords_2d
                        if len(coords_2d.shape) > 1:
                            if coords_2d.shape[1] >= 2:  # Shape is (N, 2+)
                                com_x = np.sum(coords_2d[:,0]*group_w)/wsum
                                com_y = np.sum(coords_2d[:,1]*group_w)/wsum
                            elif coords_2d.shape[0] == 2:  # Shape is (2, N)
                                com_x = np.sum(coords_2d[0]*group_w)/wsum
                                com_y = np.sum(coords_2d[1]*group_w)/wsum
                            else:
                                com_x = np.mean(coords_2d.flatten())
                                com_y = 0  # Fallback
                        else:
                            com_x = np.mean(coords_2d)
                            com_y = 0  # Fallback
                    else:
                        # Similar handling for zero weights
                        if len(coords_2d.shape) > 1:
                            if coords_2d.shape[1] >= 2:
                                com_x = coords_2d[:,0].mean()
                                com_y = coords_2d[:,1].mean()
                            elif coords_2d.shape[0] == 2:
                                com_x = coords_2d[0].mean()
                                com_y = coords_2d[1].mean()
                            else:
                                com_x = np.mean(coords_2d.flatten())
                                com_y = 0
                        else:
                            com_x = np.mean(coords_2d)
                            com_y = 0

                # Handle the checkpoint predictions similarly
                for ckpt_idx, preds_tensor in enumerate(predictions_list, start=1):
                    preds_np = preds_tensor.reshape(-1, preds_tensor.shape[-1])
                    # Project to 2D
                    if use_pca:
                        preds_pca = apply_pca_projection(preds_np, mu, comps)
                        if pca_dims>2 and inds is not None:
                            preds_2d = preds_pca[:, inds]
                        else:
                            preds_2d = preds_pca[:, :2]
                    else:
                        if project_to_simplex and base_dim==3:
                            preds_2d = _project_to_simplex(preds_np)
                        elif inds is not None:
                            preds_2d = preds_np[:, inds]
                        else:
                            preds_2d = preds_np[:, :2]

                    com_preds_x = []
                    com_preds_y = []
                    group_colors = []
                    for b_key, idxs in unique_dict.items():
                        group_w = nn_probs_np[idxs]
                        coords_2d = preds_2d[idxs]
                        wsum = group_w.sum()
                        if wsum>0:
                            com_x = np.sum(coords_2d[:,0]*group_w)/wsum
                            com_y = np.sum(coords_2d[:,1]*group_w)/wsum
                        else:
                            com_x = coords_2d[:,0].mean()
                            com_y = coords_2d[:,1].mean()
                        
                        c = color_array[idxs[0]]
                        r,g,b,a = c
                        group_colors.append(f"rgba({int(r)},{int(g)},{int(b)},{a:.2f})")
                        com_preds_x.append(com_x)
                        com_preds_y.append(com_y)

                    fig.add_trace(
                        go.Scattergl(
                            x=com_preds_x, y=com_preds_y,
                            mode='markers',
                            marker=dict(color=group_colors, size=vis_params["pred_size"]*1.5),
                            name=f"Checkpoint (COM)"
                        ),
                        row=row_idx, col=ckpt_idx+1
                    )

        else:
            # ------------------
            # Big data -> Datashader in each cell
            # ------------------
            # 1) Ground Truth in col=1
            nn_beliefs_2d_np = safe_to_numpy(nn_beliefs_2d)
            
            # Handle different shapes that might result from projections
            if project_to_simplex and base_dim == 3 and not use_pca:
                # Check if nn_beliefs_2d_np is a tuple (which happens when using project_to_simplex)
                if isinstance(nn_beliefs_2d_np, tuple) and len(nn_beliefs_2d_np) == 2:
                    # If it's a tuple from simplex projection, use each element directly
                    xvals_gt = nn_beliefs_2d_np[0]
                    yvals_gt = nn_beliefs_2d_np[1]
                elif isinstance(nn_beliefs_2d_np, np.ndarray):
                    if len(nn_beliefs_2d_np.shape) > 1 and nn_beliefs_2d_np.shape[0] == 2:
                        # If it's a 2 x N array (transposed simplex projection), use as is
                        xvals_gt = nn_beliefs_2d_np[0]
                        yvals_gt = nn_beliefs_2d_np[1]
                    else:
                        # Otherwise assume it's N x 2
                        xvals_gt = nn_beliefs_2d_np[:, 0]
                        yvals_gt = nn_beliefs_2d_np[:, 1]
                else:
                    # Fallback for unexpected type
                    print(f"Warning: Unexpected type for nn_beliefs_2d_np: {type(nn_beliefs_2d_np)}")
                    xvals_gt = np.array([0])
                    yvals_gt = np.array([0])
            else:
                # Standard case for non-simplex or PCA projections
                if isinstance(nn_beliefs_2d_np, np.ndarray):
                    if len(nn_beliefs_2d_np.shape) > 1:
                        xvals_gt = nn_beliefs_2d_np[:, 0]
                        yvals_gt = nn_beliefs_2d_np[:, 1]
                    else:
                        # Fallback if somehow we got a flat array
                        xvals_gt = nn_beliefs_2d_np
                        yvals_gt = np.zeros_like(xvals_gt)
                else:
                    # Fallback for unexpected type
                    print(f"Warning: Unexpected type for nn_beliefs_2d_np: {type(nn_beliefs_2d_np)}")
                    xvals_gt = np.array([0])
                    yvals_gt = np.array([0])

            img_gt, (x_min_gt, x_max_gt, y_min_gt, y_max_gt) = datashader_rasterize(
                xvals_gt, yvals_gt, R_int, G_int, B_int, alpha_vals
            )
            fig.add_trace(
                go.Image(
                    z=img_gt,
                    x0=x_min_gt,
                    y0=y_min_gt,
                    dx=(x_max_gt - x_min_gt)/img_gt.shape[1],
                    dy=(y_max_gt - y_min_gt)/img_gt.shape[0],
                ),
                row=row_idx, col=1
            )
            # We'll invert the y-axis so top stays top:
            fig.update_yaxes(autorange='reversed', row=row_idx, col=1)

            # 2) Each checkpoint in columns 2..n
            for ckpt_idx, preds_tensor in enumerate(predictions_list, start=1):
                preds_np = preds_tensor.reshape(-1, preds_tensor.shape[-1])
                # Project
                if use_pca:
                    preds_np_safe = safe_to_numpy(preds_np)
                    preds_pca = apply_pca_projection(preds_np_safe, mu, comps)
                    if pca_dims>2 and inds is not None:
                        preds_2d = preds_pca[:, inds]
                    else:
                        preds_2d = preds_pca[:, :2]
                else:
                    if project_to_simplex and base_dim==3:
                        # Make sure we're working with tensors for _project_to_simplex
                        if not isinstance(preds_tensor, torch.Tensor):
                            preds_tensor_torch = torch.tensor(preds_tensor)
                        else:
                            preds_tensor_torch = preds_tensor
                        preds_2d = safe_to_numpy(_project_to_simplex(preds_tensor_torch.reshape(-1, 3)))
                    elif inds is not None:
                        preds_np_safe = safe_to_numpy(preds_np)
                        preds_2d = preds_np_safe[:, inds]
                    else:
                        preds_np_safe = safe_to_numpy(preds_np)
                        preds_2d = preds_np_safe[:, :2]

                # Similar handling for prediction x and y values
                preds_2d_np = safe_to_numpy(preds_2d)
                if project_to_simplex and base_dim==3 and not use_pca:
                    # Check if preds_2d_np is a tuple (which happens when using project_to_simplex)
                    if isinstance(preds_2d_np, tuple) and len(preds_2d_np) == 2:
                        # If it's a tuple from simplex projection, use each element directly
                        xvals_p = preds_2d_np[0]
                        yvals_p = preds_2d_np[1]
                    elif isinstance(preds_2d_np, np.ndarray):
                        if len(preds_2d_np.shape) > 1 and preds_2d_np.shape[0] == 2:
                            xvals_p = preds_2d_np[0]
                            yvals_p = preds_2d_np[1]
                        else:
                            xvals_p = preds_2d_np[:, 0]
                            yvals_p = preds_2d_np[:, 1]
                    else:
                        # Fallback for unexpected type
                        print(f"Warning: Unexpected type for preds_2d_np: {type(preds_2d_np)}")
                        xvals_p = np.array([0])
                        yvals_p = np.array([0])
                else:
                    if isinstance(preds_2d_np, np.ndarray):
                        if len(preds_2d_np.shape) > 1:
                            xvals_p = preds_2d_np[:, 0]
                            yvals_p = preds_2d_np[:, 1]
                        else:
                            xvals_p = preds_2d_np
                            yvals_p = np.zeros_like(xvals_p)
                    else:
                        # Fallback for unexpected type
                        print(f"Warning: Unexpected type for preds_2d_np: {type(preds_2d_np)}")
                        xvals_p = np.array([0])
                        yvals_p = np.array([0])

                img_p, (x_min_p, x_max_p, y_min_p, y_max_p) = datashader_rasterize(
                    xvals_p, yvals_p, R_int, G_int, B_int, alpha_vals
                )
                fig.add_trace(
                    go.Image(
                        z=img_p,
                        x0=x_min_p,
                        y0=y_min_p,
                        dx=(x_max_p - x_min_p)/img_p.shape[1],
                        dy=(y_max_p - y_min_p)/img_p.shape[0],
                    ),
                    row=row_idx, col=ckpt_idx+1
                )
                fig.update_yaxes(autorange='reversed', row=row_idx, col=ckpt_idx+1)

        # Hide axis lines/ticks for this row
        for col_idx in range(1, n_cols+1):
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title_text="",
                row=row_idx, col=col_idx
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title_text="",
                row=row_idx, col=col_idx
            )

        # Add a layer label on the left side
        display_name = layer_name
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."
        fig.add_annotation(
            x=-0.1,
            y=0.5,
            text=display_name,
            showarrow=False,
            font=dict(size=11),
            xref="x domain",
            yref="y domain",
            xanchor="right",
            yanchor="middle",
            row=row_idx, col=1,
            textangle=270
        )

        row_idx += 1

    # Now add top row titles for ground-truth + each checkpoint
    title_font = dict(size=10)

    # label for col=1 (ground truth)
    fig.add_annotation(
        x=0.5,
        y=1.05,
        text="Ground Truth",
        showarrow=False,
        font=title_font,
        xref="x domain",
        yref="y domain",
        xanchor="center",
        yanchor="bottom",
        row=1, col=1
    )

    for col_idx, ckpt_name in enumerate(checkpoint_names[:max_checkpoints], start=1):
        title_html = ckpt_name.replace("\n","<br>")
        fig.add_annotation(
            x=0.5,
            y=1.05,
            text=title_html,
            showarrow=False,
            font=title_font,
            xref="x domain",
            yref="y domain",
            xanchor="center",
            yanchor="bottom",
            row=1, col=col_idx+1
        )

    # Add sweep/run in the top-left corner (paper coords)
    if sweep_id and run_id:
        fig.add_annotation(
            x=0.01,
            y=0.99,
            text=f"Sweep: {sweep_id}<br>Run: {run_id}",
            showarrow=False,
            font=dict(size=9, color="gray"),
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left"
        )

    return fig

def create_visualizations_datashader(
    nn_beliefs, nn_probs,
    predictions_list, 
    checkpoint_names,
    sweep_id=None,
    run_id=None, 
    ground_truth_size=2,  # Not actually used for raster, but keep the signature
    prediction_size=1.5,
    project_to_simplex=True,
    inds=None,
    coms=False,
    use_pca=False,
    title_suffix=""
):
    """
    Datashader-based version of create_visualizations(...), used only if run_id
    contains 'tom_quantum'. We do nearly the same subplot layout, but each subplot
    gets a rasterized image instead of a giant Scattergl trace.
    
    This tries to keep the *same color logic* by summing up R*alpha, G*alpha, B*alpha
    and alpha in each pixel, then dividing to get average color. 
    """
    
    from plotly.subplots import make_subplots
    from epsilon_transformers.visualization.plots import _project_to_simplex
    
    # Convert beliefs, probabilities to numpy
    nn_beliefs_np = nn_beliefs.reshape(-1, nn_beliefs.shape[-1])
    nn_probs_np = nn_probs.reshape(-1)
    
    # Handle color/alpha arrays the same way we do in create_visualizations
    base_dim = nn_beliefs_np.shape[-1]
    
    # In principle, you'd reuse your existing color logic:
    if base_dim == 3:
        # Reuse your function
        _, _, colors, (R_int, G_int, B_int) = prepare_visualization_data(nn_beliefs, nn_probs)
        alpha_vals = [float(c.split(',')[-1].replace(')', '')) for c in colors]  # parse from "rgba(r,g,b,a)"
    else:
        # For dim>3, you do PCA for color
        from sklearn.decomposition import PCA
        pcs_3 = PCA(n_components=3).fit_transform(nn_beliefs_np)
        min_vals = pcs_3.min(axis=0)
        max_vals = pcs_3.max(axis=0)
        range_vals = (max_vals - min_vals) + 1e-8
        alpha_vals = compute_alpha_float(nn_probs_np, scale=1, min_alpha=0.15)
        
        R_int, G_int, B_int = [], [], []
        for i in range(len(pcs_3)):
            r = int(255 * (pcs_3[i,0] - min_vals[0]) / range_vals[0])
            g = int(255 * (pcs_3[i,1] - min_vals[1]) / range_vals[1])
            b = int(255 * (pcs_3[i,2] - min_vals[2]) / range_vals[2])
            R_int.append(r); G_int.append(g); B_int.append(b)
    
    R_int = np.array(R_int, dtype=float)
    G_int = np.array(G_int, dtype=float)
    B_int = np.array(B_int, dtype=float)
    
    # If alpha_vals didn't exist (we only got it for base_dim==3 above), compute it:
    if base_dim != 3:
        alpha_vals = compute_alpha_float(nn_probs_np, scale=1, min_alpha=0.15)
    alpha_vals = np.array(alpha_vals, dtype=float)

    # Next, decide on the 2D projection for the ground truth
    if use_pca:
        mu, components = weighted_pca(nn_beliefs_np, nn_probs_np, n_components=2)
        nn_beliefs_2d = apply_pca_projection(nn_beliefs_np, mu, components)
    else:
        if project_to_simplex and base_dim == 3:
            nn_beliefs_2d = _project_to_simplex(nn_beliefs.reshape(-1, 3))
        elif inds is not None:
            nn_beliefs_2d = nn_beliefs_np[:, inds]
        else:
            nn_beliefs_2d = nn_beliefs_np[:, :2]
    
    # We'll also project each checkpoint's predictions. We'll store them in a list
    all_preds_2d = []
    for predictions in predictions_list:
        preds_np = predictions.reshape(-1, predictions.shape[-1]).detach().cpu().numpy()
        if use_pca:
            preds_2d = apply_pca_projection(preds_np, mu, components)
        else:
            if project_to_simplex and base_dim == 3:
                preds_2d = _project_to_simplex(predictions.reshape(-1, 3))
            elif inds is not None:
                preds_2d = preds_np[:, inds]
            else:
                preds_2d = preds_np[:, :2]
        all_preds_2d.append(preds_2d)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, 
        cols=len(predictions_list) + 1  # +1 for ground truth
    )
    
    # For the sake of "close to same," let's define a function to rasterize points
    def datashader_rasterize(xvals, yvals, R, G, B, A, plot_width=300, plot_height=300):
        """
        Sums up R*A, G*A, B*A, and A in each pixel, then divides to get final color.
        Returns: (img, (xrange, yrange)) so we can place it in Plotly go.Image
        """
        df = pd.DataFrame({
            'x': xvals,
            'y': yvals,
            'RA': R*A,  # We'll sum these
            'GA': G*A,
            'BA': B*A,
            'A': A
        })
        
        # Decide bounding box from the data
        x_min, x_max = np.min(xvals), np.max(xvals)
        y_min, y_max = np.min(yvals), np.max(yvals)
        if x_min == x_max:
            # avoid zero-range
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        
        cvs = ds.Canvas(
            plot_width=plot_width, 
            plot_height=plot_height,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max)
        )
        agg_R = cvs.points(df, 'x', 'y', ds.sum('RA'))
        agg_G = cvs.points(df, 'x', 'y', ds.sum('GA'))
        agg_B = cvs.points(df, 'x', 'y', ds.sum('BA'))
        agg_A = cvs.points(df, 'x', 'y', ds.sum('A'))
        
        # Convert xarray to numpy
        r_vals = agg_R.values
        g_vals = agg_G.values
        b_vals = agg_B.values
        a_vals = agg_A.values
        
        # Weighted average color in each pixel
        with np.errstate(divide='ignore', invalid='ignore'):
            r_final = r_vals / a_vals
            g_final = g_vals / a_vals
            b_final = b_vals / a_vals
        # Where a_vals == 0, fill with 0 or white
        r_final[np.isnan(r_final)] = 0
        g_final[np.isnan(g_final)] = 0
        b_final[np.isnan(b_final)] = 0
        
        # Clip color channels
        r_final = np.clip(r_final, 0, 255)
        g_final = np.clip(g_final, 0, 255)
        b_final = np.clip(b_final, 0, 255)
        
        # The alpha we "display" can be scaled 0..1, but let's cap it at 1
        # If you want the average alpha, do a_vals / count. But here we'll treat
        # it as "1" whenever there is any data. Or you can do a ratio to 1:
        # a_final = np.clip(a_vals, 0, 1)
        # but that can look too faint if you have many small alphas.
        # Let's do a simpler approach: if sum of alpha > 0, we set alpha=1
        a_final = (a_vals > 0).astype(np.float32)
        
        # Stack into RGBA image: shape (height, width, 4)
        height, width = r_vals.shape
        rgba_img = np.ones((height, width, 4), dtype=np.uint8) * 255
        rgba_img[..., 0] = r_final.astype(np.uint8)
        rgba_img[..., 1] = g_final.astype(np.uint8)
        rgba_img[..., 2] = b_final.astype(np.uint8)
        rgba_img[..., 3] = (a_final*255).astype(np.uint8)
        
        # Return the image array + the bounding box
        return rgba_img, (x_min, x_max, y_min, y_max)
    
    # If coms=True, handle that in a simpler way (COM is just a small set)
    # so we can just do normal scatter for COM. Otherwise, we do raster per subplot.
    if coms:
        # The logic for center-of-mass is the same as your normal function:
        # We'll do normal Scattergl for these small sets. 
        # (Hence we skip big Datashader for coms because it's presumably few points.)
        
        # ...................
        # Identical COM code
        # ...................
        from collections import defaultdict
        unique_dict = defaultdict(list)
        for i, b in enumerate(nn_beliefs_np):
            key = tuple(np.round(b, 5))
            unique_dict[key].append(i)
        
        # Ground-truth COM
        com_ground_x = []
        com_ground_y = []
        com_colors = []
        
        # We can just call your original color array from above (colors) 
        # to color each group by the first index
        # If base_dim==3, we had a direct array. If not, we built them in PCA. 
        # We'll store them in the same 'colors' list (but we only have them 
        # if base_dim==3 or we did the PCA approach for color).
        # Actually, let's just do a second pass to build a color array for each group:
        color_array = np.array([
            (R_int[i], G_int[i], B_int[i], alpha_vals[i]) for i in range(len(R_int))
        ])
        
        # Project nn_beliefs_2d is shape (N, 2)
        for b_key, idx_list in unique_dict.items():
            group_w = nn_probs_np[idx_list]
            coords_2d = nn_beliefs_2d[idx_list]
            wsum = group_w.sum()
            if wsum > 0:
                com_x = np.sum(coords_2d[:, 0] * group_w) / wsum
                com_y = np.sum(coords_2d[:, 1] * group_w) / wsum
            else:
                com_x = coords_2d[:, 0].mean()
                com_y = coords_2d[:, 1].mean()
            com_ground_x.append(com_x)
            com_ground_y.append(com_y)
            
            # Just pick the color of the first index
            c = color_array[idx_list[0]]
            # c is (R, G, B, alpha)
            r, g, b, a = c
            com_colors.append(f"rgba({int(r)},{int(g)},{int(b)},{a:.2f})")

        # Plot ground truth in col=1
        gt_trace = go.Scattergl(
            x=com_ground_x,
            y=com_ground_y,
            mode='markers',
            marker=dict(
                color=com_colors,
                size=ground_truth_size*1.5,
                opacity=1
            ),
            name="Ground Truth (COM)"
        )
        fig.add_trace(gt_trace, row=1, col=1)

        # For each checkpoint
        for i, preds_2d in enumerate(all_preds_2d, start=1):
            com_preds_x = []
            com_preds_y = []
            group_colors = []
            for b_key, idx_list in unique_dict.items():
                group_w = nn_probs_np[idx_list]
                coords_2d = preds_2d[idx_list]
                wsum = group_w.sum()
                if wsum > 0:
                    com_x = np.sum(coords_2d[:, 0] * group_w) / wsum
                    com_y = np.sum(coords_2d[:, 1] * group_w) / wsum
                else:
                    com_x = coords_2d[:, 0].mean()
                    com_y = coords_2d[:, 1].mean()
                
                c = color_array[idx_list[0]]
                r, g, b, a = c
                group_colors.append(f"rgba({int(r)},{int(g)},{int(b)},{a:.2f})")
                
                com_preds_x.append(com_x)
                com_preds_y.append(com_y)
            
            pred_trace = go.Scattergl(
                x=com_preds_x,
                y=com_preds_y,
                mode='markers',
                marker=dict(
                    color=group_colors,
                    size=prediction_size*1.5,
                    opacity=1
                ),
                name=f"Checkpoint {i} (COM)"
            )
            fig.add_trace(pred_trace, row=1, col=i+1)
    
    else:
        # --------------- 
        # Normal big scatter => do a Datashader raster in each subplot
        # ---------------
        
        # 1) Ground Truth
        # If your 2D is shape(N,2) then x=[:,0], y=[:,1]
        if project_to_simplex and base_dim == 3 and not use_pca:
            xvals_gt = nn_beliefs_2d[0]
            yvals_gt = nn_beliefs_2d[1]
        else:
            xvals_gt = nn_beliefs_2d[:,0]
            yvals_gt = nn_beliefs_2d[:,1]
        
        # Rasterize
        img_gt, (x_min_gt, x_max_gt, y_min_gt, y_max_gt) = datashader_rasterize(
            xvals_gt, yvals_gt, R_int, G_int, B_int, alpha_vals
        )
        
        # Add it as go.Image in col=1
        fig.add_trace(
            go.Image(
                z=img_gt,
                x0=x_min_gt,
                y0=y_min_gt,
                dx=(x_max_gt - x_min_gt)/img_gt.shape[1],
                dy=(y_max_gt - y_min_gt)/img_gt.shape[0],
            ),
            row=1, col=1
        )
        
        # 2) Each checkpoint
        for i, preds_2d in enumerate(all_preds_2d, start=1):
            if project_to_simplex and base_dim == 3 and not use_pca:
                xvals_p = preds_2d[0]
                yvals_p = preds_2d[1]
            else:
                xvals_p = preds_2d[:,0]
                yvals_p = preds_2d[:,1]
            
            img_p, (x_min_p, x_max_p, y_min_p, y_max_p) = datashader_rasterize(
                xvals_p, yvals_p,
                R_int, G_int, B_int, alpha_vals
            )
            
            fig.add_trace(
                go.Image(
                    z=img_p,
                    x0=x_min_p,
                    y0=y_min_p,
                    dx=(x_max_p - x_min_p)/img_p.shape[1],
                    dy=(y_max_p - y_min_p)/img_p.shape[0],
                ),
                row=1, col=i+1
            )
    
    # -----------------------
    # Match your subplot styling, etc.
    # -----------------------
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=5, r=5, t=25, b=30),
        height=175,
        width=150 * (len(predictions_list) + 1)
    )

    # Hide axis lines/ticks
    for col_i in range(1, len(predictions_list) + 2):
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_text="",
            row=1, col=col_i
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_text="",
            row=1, col=col_i
        )
        # Because images in Plotly are usually top-down, you may want:
        fig.update_yaxes(autorange="reversed", row=1, col=col_i)

    # Title annotations, same logic as your create_visualizations
    title_font = dict(size=10)
    title_y = 0.
    col_width = 1.0 / (len(predictions_list) + 1)
    centers = [(i + 0.5) * col_width for i in range(len(predictions_list) + 1)]
    move_outwards = 0.015
    original_min, original_max = centers[0], centers[-1]
    new_min, new_max = original_min - move_outwards, original_max + move_outwards
    centers = [
        new_min + (new_max - new_min) * i / (len(centers) - 1) 
        for i in range(len(centers))
    ]

    gt_title = "Ground Truth"
    if title_suffix:
        gt_title += f" ({title_suffix})"
        
    fig.add_annotation(
        x=centers[0],
        y=title_y,
        text=gt_title,
        showarrow=False,
        font=title_font,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        align="center"
    )

    for i, title in enumerate(checkpoint_names, start=1):
        title_html = title.replace("\n", "<br>")
        fig.add_annotation(
            x=centers[i],
            y=title_y,
            text=title_html,
            showarrow=False,
            font=title_font,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            align="center"
        )

    # Possibly annotate the top-left corner with sweep/run
    if sweep_id and run_id:
        info_text = f"Sweep: {sweep_id}<br>Run: {run_id}"
        if title_suffix:
            info_text += f"<br>{title_suffix}"
        fig.add_annotation(
            x=0.0,
            y=1.0,
            text=info_text,
            showarrow=False,
            font=dict(size=9, color="gray"),
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            align="left"
        )

    return fig

# ------------------------------
# Weighted PCA Utilities
# ------------------------------
def weighted_pca(data: np.ndarray, weights: np.ndarray, n_components=2):
    """
    Computes a weighted PCA of 'data' using 'weights' (>= 0).
    Returns (mean, components), where:
      - mean is shape (D,)
      - components is shape (D, n_components)
    """
    assert data.shape[0] == weights.shape[0], "Data and weight size mismatch."

    # Normalize weights so they sum to 1 (for convenience)
    wsum = weights.sum()
    if wsum <= 0:
        # Fallback: if all weights are zero or negative (shouldn't happen),
        # just do an unweighted PCA.
        wnorm = np.ones_like(weights) / len(weights)
    else:
        wnorm = weights / wsum

    # Weighted mean
    mu = np.average(data, axis=0, weights=wnorm)  # shape (D,)

    # Subtract mean
    centered = data - mu

    # Weighted covariance via sqrt(weights)
    sqrt_w = np.sqrt(wnorm)[:, None]  # shape (N, 1)
    weighted_centered = centered * sqrt_w  # shape (N, D)
    cov = weighted_centered.T @ weighted_centered  # shape (D, D)

    # SVD of cov
    U, S, Vt = np.linalg.svd(cov, full_matrices=False)
    # U[:, :n_components] are our principal directions
    components = U[:, :n_components]  # shape (D, n_components)

    return mu, components

def apply_pca_projection(data: np.ndarray, mu: np.ndarray, components: np.ndarray):
    """
    Projects data (shape (N, D)) into 2D via the principal components.
    data_2d = (data - mu) @ components
    """
    centered = data - mu
    return centered @ components


# ------------------------------
# Visualization Helpers
# ------------------------------
def compute_alpha_float(prob, scale=10, min_alpha=0.2):
    """
    Compute an alpha value (float in [0,1]) from a probability value (0-1).
    Uses a logarithmic transformation so that even very low probabilities are
    boosted to at least min_alpha.
    """
    transformed = np.log1p(prob * scale) / np.log1p(scale)
    alpha = min_alpha + (1 - min_alpha) * transformed
    return np.clip(alpha, 0, 1)

def prepare_visualization_data(nn_beliefs, nn_probs, min_alpha=0.15):
    """
    Prepare data for visualization, including color encoding based on R/G/B.
    This code is left mostly unchanged from your original snippet—just be mindful
    it's not used if we do Weighted PCA. We'll still call it if we
    want an RGB color for each point, or you can adapt further.
    
    Args:
        nn_beliefs: Belief states tensor
        nn_probs: Probability weights tensor
        min_alpha: Minimum alpha value (transparency) for the scatter points
    """
    # Make sure inputs are numpy arrays
    if not isinstance(nn_beliefs, np.ndarray):
        if isinstance(nn_beliefs, torch.Tensor):
            nn_beliefs_np = nn_beliefs.detach().cpu().numpy()
        else:
            nn_beliefs_np = np.array(nn_beliefs)
    else:
        nn_beliefs_np = nn_beliefs
    
    if not isinstance(nn_probs, np.ndarray):
        if isinstance(nn_probs, torch.Tensor):
            nn_probs_np = nn_probs.detach().cpu().numpy()
        else:
            nn_probs_np = np.array(nn_probs)
    else:
        nn_probs_np = nn_probs
        
    # Get the shapes right
    belief_dim = nn_beliefs_np.shape[-1]
    nn_beliefs_reshaped = nn_beliefs_np.reshape(-1, belief_dim)
    nn_probs_reshaped = nn_probs_np.reshape(-1)
    nn_probs_reshaped = nn_probs_reshaped / (nn_probs_reshaped.sum() + 1e-10)  # Add epsilon to avoid division by zero

    x = nn_beliefs_reshaped[:, 0]
    y = nn_beliefs_reshaped[:, 1]
    z = nn_beliefs_reshaped[:, 2]

    B_val = np.sqrt(x**2 + y**2)

    epsilon = 1e-8
    min_R, max_R = x.min(), x.max()
    min_G, max_G = y.min(), y.max()
    min_B, max_B = B_val.min(), B_val.max()

    R_norm = (x - min_R) / (max_R - min_R + epsilon)
    G_norm = (y - min_G) / (max_G - min_G + epsilon)
    B_norm = (B_val - min_B) / (max_B - min_B + epsilon)

    R_int = (R_norm * 255).astype(int)
    G_int = (G_norm * 255).astype(int)
    B_int = (B_norm * 255).astype(int)

    alpha_float = compute_alpha_float(nn_probs_reshaped, scale=1, min_alpha=min_alpha)
    colors = [
        f"rgba({r},{g},{b},{alpha:.2f})" 
        for r, g, b, alpha in zip(R_int, G_int, B_int, alpha_float)
    ]
    
    return nn_beliefs_reshaped, nn_probs_reshaped, colors, (R_int, G_int, B_int)


def create_visualizations(
    nn_beliefs, nn_probs,
    predictions_list, 
    checkpoint_names,
    sweep_id=None,
    run_id=None, 
    ground_truth_size=2,
    prediction_size=1.5,
    project_to_simplex=True,
    inds=None,
    coms=False,
    use_pca=False,
    title_suffix=""
):
    """
    Create a plotly figure with ground truth and multiple checkpoint predictions.
    If `use_pca=True`, we do Weighted PCA for all beliefs (and predictions).
    If `coms=True`, we group points by unique ground-truth beliefs and plot centers of mass.
    
    Args:
        title_suffix: An optional string to append to the plot title (e.g., layer name)
    """

    # Convert to numpy
    nn_beliefs_np = nn_beliefs.reshape(-1, nn_beliefs.shape[-1]).detach().cpu().numpy()
    nn_probs_np = nn_probs.reshape(-1).detach().cpu().numpy()
    # We'll also get colors for each point (based on R/G in 3D) for styling
    # If your dimension > 3, you might want to adapt or skip this
    # (Here we do it for the sake of consistent coloring)
    base_dim = nn_beliefs_np.shape[-1]
    print(f"Base dimension: {base_dim}")

    # Color and alpha
    # This uses the original "prepare_visualization_data" which expects 3D beliefs
    # If you have more than 3 dims, adapt or skip. For demonstration, we'll do:
    if base_dim == 3:
        _, _, colors, _ = prepare_visualization_data(nn_beliefs, nn_probs)
    else:
        from sklearn.decomposition import PCA

        def pca_for_color(beliefs_np, n_components=3):
            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(beliefs_np)  # shape (N, 3)
            return pcs

        pcs_3d = pca_for_color(nn_beliefs_np, 3)
        min_vals = pcs_3d.min(axis=0)
        max_vals = pcs_3d.max(axis=0)
        range_vals = (max_vals - min_vals) + 1e-8

        colors = []
        alpha_vals = compute_alpha_float(nn_probs_np, scale=1, min_alpha=0.15)
        for i in range(len(pcs_3d)):
            r = int(255 * (pcs_3d[i,0] - min_vals[0]) / range_vals[0])
            g = int(255 * (pcs_3d[i,1] - min_vals[1]) / range_vals[1])
            b = int(255 * (pcs_3d[i,2] - min_vals[2]) / range_vals[2])
            a = alpha_vals[i]
            colors.append(f"rgba({r},{g},{b},{a:.2f})")

    # Weighted PCA if requested
    if use_pca:
        mu, components = weighted_pca(nn_beliefs_np, nn_probs_np, n_components=2)
        # Project ground truth
        nn_beliefs_2d = apply_pca_projection(nn_beliefs_np, mu, components)
    else:
        # Otherwise, do existing logic
        if project_to_simplex and base_dim == 3:
            nn_beliefs_2d = _project_to_simplex(nn_beliefs.reshape(-1, 3))
        elif inds is not None:
            nn_beliefs_2d = nn_beliefs_np[:, inds]
        else:
            # default to first two dims
            nn_beliefs_2d = nn_beliefs_np[:, :2]

    # We'll store the 2D predictions for each checkpoint
    all_preds_2d = []
    for predictions in predictions_list:
        preds_np = predictions.reshape(-1, predictions.shape[-1])

        if use_pca:
            preds_2d = apply_pca_projection(preds_np, mu, components)
        else:
            if project_to_simplex and base_dim == 3:
                preds_2d = _project_to_simplex(preds_np)
            elif inds is not None:
                preds_2d = preds_np[:, inds]
            else:
                preds_2d = preds_np[:, :2]

        all_preds_2d.append(preds_2d)

    # Create subplot figure
    fig = make_subplots(
        rows=1, 
        cols=len(predictions_list) + 1  # +1 for ground truth
    )

    # If not coms, we just scatter all the raw points
    if not coms:
        # Plot ground truth beliefs (2D)
        if project_to_simplex and base_dim == 3:
            x_beliefs, y_beliefs = nn_beliefs_2d[0], nn_beliefs_2d[1]
        else:
            x_beliefs, y_beliefs = nn_beliefs_2d[:, 0], nn_beliefs_2d[:, 1]
        beliefs_trace = go.Scattergl(
            x=x_beliefs,
            y=y_beliefs,
            mode='markers',
            marker=dict(
                color=colors,
                size=ground_truth_size
            ),
            name="Belief States"
        )
        fig.add_trace(beliefs_trace, row=1, col=1)

        # Add predictions from each checkpoint
        for i, (preds_2d) in enumerate(all_preds_2d, start=1):
            if project_to_simplex and base_dim == 3:
                x_preds, y_preds = preds_2d[0], preds_2d[1]
            else:
                x_preds, y_preds = preds_2d[:, 0], preds_2d[:, 1]
            pred_trace = go.Scattergl(
                x=x_preds,
                y=y_preds,
                mode='markers',
                marker=dict(
                    color=colors,  # same color mapping
                    size=prediction_size
                ),
                name=f"Checkpoint {i}"
            )
            fig.add_trace(pred_trace, row=1, col=i+1)

    else:
        # -----------------------
        # COM logic
        # -----------------------
        # 1) Group ground truth by unique beliefs (rounded to avoid float mismatch)
        unique_dict = defaultdict(list)
        for i, b in enumerate(nn_beliefs_np):
            # Round to handle float comparison
            key = tuple(np.round(b, 5))
            unique_dict[key].append(i)

        # 2) For each group, compute the weighted COM in the ground truth 2D space
        com_ground_x = []
        com_ground_y = []
        com_colors = []
        for b_key, idx_list in unique_dict.items():
            group_w = nn_probs_np[idx_list]
            coords_2d = nn_beliefs_2d[idx_list]  # shape (Ngroup, 2)

            wsum = group_w.sum()
            if wsum > 0:
                com_x = np.sum(coords_2d[:, 0] * group_w) / wsum
                com_y = np.sum(coords_2d[:, 1] * group_w) / wsum
            else:
                # fallback
                com_x = coords_2d[:, 0].mean()
                com_y = coords_2d[:, 1].mean()

            com_ground_x.append(com_x)
            com_ground_y.append(com_y)
            # For color, just take the color of the first index
            com_colors.append(colors[idx_list[0]])

        # Plot the COMs for ground truth
        com_trace_gt = go.Scattergl(
            x=com_ground_x,
            y=com_ground_y,
            mode='markers',
            marker=dict(
                color=com_colors,
                size=ground_truth_size * 1.5,  # slightly bigger
                opacity=1
            ),
            name="Ground Truth (COM)"
        )
        fig.add_trace(com_trace_gt, row=1, col=1)

        # 3) For each checkpoint's predictions, compute group COM
        for i, preds_2d in enumerate(all_preds_2d, start=1):
            com_preds_x = []
            com_preds_y = []
            group_colors = []
            for b_key, idx_list in unique_dict.items():
                group_w = nn_probs_np[idx_list]
                coords_2d = preds_2d[idx_list]

                wsum = group_w.sum()
                if wsum > 0:
                    com_x = np.sum(coords_2d[:, 0] * group_w) / wsum
                    com_y = np.sum(coords_2d[:, 1] * group_w) / wsum
                else:
                    com_x = coords_2d[:, 0].mean()
                    com_y = coords_2d[:, 1].mean()

                com_preds_x.append(com_x)
                com_preds_y.append(com_y)
                group_colors.append(colors[idx_list[0]])

            com_trace_ckpt = go.Scattergl(
                x=com_preds_x,
                y=com_preds_y,
                mode='markers',
                marker=dict(
                    color=group_colors,
                    size=prediction_size * 1.5,
                    opacity=1
                ),
                name=f"Checkpoint {i} (COM)"
            )
            fig.add_trace(com_trace_ckpt, row=1, col=i+1)

    # -----------------------
    # Style each subplot
    # -----------------------
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=5, r=5, t=25, b=30),
        height=175,
        width=150 * (len(predictions_list) + 1)
    )

    # Hide all axis lines, ticks, labels
    for col_i in range(1, len(predictions_list) + 2):
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_text="",
            row=1, col=col_i
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_text="",
            row=1, col=col_i
        )

    # Subplot titles
    title_font = dict(size=10)
    title_y = 0.
    col_width = 1.0 / (len(predictions_list) + 1)
    centers = [(i + 0.5) * col_width for i in range(len(predictions_list) + 1)]
    move_outwards = 0.015
    original_min, original_max = centers[0], centers[-1]
    new_min, new_max = original_min - move_outwards, original_max + move_outwards
    # Recompute centers:
    centers = [new_min + (new_max - new_min) * i / (len(centers) - 1) for i in range(len(centers))]

    # Ground Truth annotation
    gt_title = "Ground Truth"
    if title_suffix:
        gt_title += f" ({title_suffix})"
        
    fig.add_annotation(
        x=centers[0],
        y=title_y,
        text=gt_title,
        showarrow=False,
        font=title_font,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        align="center"
    )

    # Checkpoint annotations
    for i, title in enumerate(checkpoint_names, start=1):
        title_html = title.replace("\n", "<br>")
        fig.add_annotation(
            x=centers[i],
            y=title_y,
            text=title_html,
            showarrow=False,
            font=title_font,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            align="center"
        )

    # Add sweep/run in the top-left corner
    if sweep_id and run_id:
        info_text = f"Sweep: {sweep_id}<br>Run: {run_id}"
        if title_suffix:
            info_text += f"<br>{title_suffix}"
            
        fig.add_annotation(
            x=0.0,
            y=1.0,
            text=info_text,
            showarrow=False,
            font=dict(size=9, color="gray"),
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            align="left"
        )

    return fig


# ------------------------------
# Script Main
# ------------------------------
def load_model_and_extract_beliefs(s3_loader, sweep_id, run_id, checkpoint, device="cpu"):
    """Load a model and extract beliefs from it"""
    print(f"Loading model from checkpoint: {checkpoint}")
    model, run_config = s3_loader.load_checkpoint(sweep_id, run_id, checkpoint, device=device)
    
    # Prepare neural network beliefs
    n_ctx = run_config['model_config']['n_ctx']
    run_config['n_ctx'] = n_ctx
    nn_inputs, nn_beliefs, nn_indices, nn_probs, nn_unnormalized = prepare_msp_data(run_config, run_config['process_config'])
    
    return model, run_config, nn_inputs, nn_beliefs, nn_probs

def get_predictions(model, nn_inputs, nn_beliefs, nn_probs, device="cpu", layer_key="ln_final.hook_normalized"):
    """Extract activations and predict beliefs for a specific layer activation"""
    act_extractor = ActivationExtractor(device)
    nn_acts = act_extractor.extract_activations(model, nn_inputs, "transformer", TRANSFORMER_ACTIVATION_KEYS)
    
    regression_analyzer = RegressionAnalyzer(device=device, use_efficient_pinv=True)
    # The user mentioned the regression 'already does so with the weights'.
    # Make sure run_single_rcond_sweep_with_predictions is indeed using nn_probs as sample weights internally.
    best_results = run_single_rcond_sweep_with_predictions(
        regression_analyzer, 
        nn_acts[layer_key],
        nn_beliefs, 
        nn_probs / nn_probs.sum(), 
        rcond_values=RCOND_SWEEP_LIST
    )
    
    return best_results, nn_acts

def get_all_layer_predictions(model, nn_inputs, nn_beliefs, nn_probs, device="cpu"):
    """
    Extract activations from all layers and run prediction for each layer individually.
    Also concatenate all layers and run prediction on the combined representation.
    """
    act_extractor = ActivationExtractor(device)
    nn_acts = act_extractor.extract_activations(model, nn_inputs, "transformer", TRANSFORMER_ACTIVATION_KEYS)
    
    regression_analyzer = RegressionAnalyzer(device=device, use_efficient_pinv=True)
    normalized_probs = nn_probs / nn_probs.sum()
    
    # Dictionary to store results for each layer
    layer_results = {}
    
    # Process each layer individually
    for layer_key in nn_acts.keys():
        print(f"Processing layer: {layer_key}")
        layer_results[layer_key] = run_single_rcond_sweep_with_predictions(
            regression_analyzer,
            nn_acts[layer_key],
            nn_beliefs,
            normalized_probs,
            rcond_values=RCOND_SWEEP_LIST
        )
    
    # Process all layers combined by concatenating their activations
    print("Processing all layers concatenated")
    
    # First, let's determine the original shape and ensure all tensors match
    # We assume that all activations should have the same first dimensions corresponding to batch and sequence
    original_shape = None
    first_dims_size = None
    all_activations = []
    
    # First pass to determine shapes and flatten activations properly
    for layer_key, acts in nn_acts.items():
        # Get the shape
        if original_shape is None:
            # Save the original shape from the first layer
            original_shape = acts.shape
            # Calculate the product of all dimensions except the last one
            first_dims_size = np.prod(original_shape[:-1])
        
        # Flatten all dimensions except the last (feature) dimension
        # This ensures all activations have shape (total_elements, feature_dim)
        flattened_acts = acts.reshape(-1, acts.shape[-1])
        
        # Verify shape consistency
        if flattened_acts.shape[0] != first_dims_size:
            print(f"Warning: Layer {layer_key} has inconsistent shape: {acts.shape}, expected first dims to flatten to {first_dims_size}")
            # Skip this layer to avoid errors
            continue
            
        all_activations.append(flattened_acts)
    
    if not all_activations:
        print("Error: No activations could be properly processed for concatenation")
        layer_results["all_layers_combined"] = None
        return layer_results, nn_acts
    
    # Concatenate all activations along the feature dimension (dim=1)
    combined_acts = torch.cat(all_activations, dim=1)
    print(f"Combined activations shape: {combined_acts.shape}")
    
    # Ensure nn_beliefs and nn_probs are properly shaped for regression
    reshaped_beliefs = nn_beliefs.reshape(-1, nn_beliefs.shape[-1])
    reshaped_probs = normalized_probs.reshape(-1)
    
    # Verify shapes match before regression
    if combined_acts.shape[0] != reshaped_beliefs.shape[0] or combined_acts.shape[0] != reshaped_probs.shape[0]:
        print(f"Error: Shape mismatch - activations: {combined_acts.shape}, beliefs: {reshaped_beliefs.shape}, probs: {reshaped_probs.shape}")
        # Make a best effort to align shapes if possible
        min_size = min(combined_acts.shape[0], reshaped_beliefs.shape[0], reshaped_probs.shape[0])
        combined_acts = combined_acts[:min_size]
        reshaped_beliefs = reshaped_beliefs[:min_size]
        reshaped_probs = reshaped_probs[:min_size]
        print(f"Adjusted shapes to: activations: {combined_acts.shape}, beliefs: {reshaped_beliefs.shape}, probs: {reshaped_probs.shape}")
    
    # Run regression on the combined activations
    combined_results = run_single_rcond_sweep_with_predictions(
        regression_analyzer,
        combined_acts,
        reshaped_beliefs,  # Use the flattened beliefs directly
        reshaped_probs,    # Use the flattened probs directly
        rcond_values=RCOND_SWEEP_LIST
    )
    
    # If the regression was successful and we have predictions, reshape them back to original shape
    if combined_results and 'predictions' in combined_results:
        # Predictions will be in flattened form, reshape to match original belief shape
        flat_preds = combined_results['predictions']
        if isinstance(flat_preds, torch.Tensor) and flat_preds.dim() == 2:
            # Reshape to match the original nn_beliefs shape
            combined_results['predictions'] = flat_preds.reshape(nn_beliefs.shape)
    
    layer_results["all_layers_combined"] = combined_results
    
    return layer_results, nn_acts

def create_consolidated_visualization(
    nn_beliefs, nn_probs,
    layer_predictions_dict,
    checkpoint_names,
    sweep_id=None,
    run_id=None,
    vis_params=None,
    layer_order=None
):
    """
    Create a single consolidated plot with multiple rows, where each row represents a different layer.
    
    Args:
        nn_beliefs: Ground truth belief states
        nn_probs: Ground truth probabilities
        layer_predictions_dict: Dictionary mapping layer names to lists of predictions
        checkpoint_names: Names of the checkpoints
        sweep_id: ID of the sweep
        run_id: ID of the run
        vis_params: Dictionary of visualization parameters
        layer_order: Optional list specifying the order in which to display layers
    
    Returns:
        Plotly figure with multiple rows, each representing a layer
    """
    if vis_params is None:
        vis_params = {
            "gt_size": 2,
            "pred_size": 1.5,
            "project_to_simplex": False,
            "inds": [0, 1],
            "coms": False,
            "use_pca": False
        }
    
    # Extract optional parameters with defaults
    pca_dims = vis_params.get("pca_dims", 2)  # Default to 2 dimensions if not specified
    min_alpha = vis_params.get("min_alpha", 0.15)  # Default alpha if not specified
    
    # Filter out any empty layers
    valid_layers = {layer: preds for layer, preds in layer_predictions_dict.items() if preds}
    
    # If layer_order is specified, use it to order the layers
    if layer_order:
        layers_to_plot = [layer for layer in layer_order if layer in valid_layers]
    else:
        # Default ordering: Put "all_layers_combined" at the end, sort the rest
        regular_layers = [layer for layer in valid_layers.keys() if layer != "all_layers_combined"]
        regular_layers.sort() # Sort alphabetically by default
        layers_to_plot = regular_layers
        if "all_layers_combined" in valid_layers:
            layers_to_plot.append("all_layers_combined")
    
    # Determine the number of checkpoint columns (plus 1 for ground truth)
    max_checkpoints = max([len(preds) for preds in valid_layers.values()], default=0)
    n_cols = max_checkpoints + 1
    n_rows = len(layers_to_plot)
    
    # Create subplots with one row per layer
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=None,  # We'll add custom annotations
        vertical_spacing=0.05,
        horizontal_spacing=0.01
    )
    
    # Calculate optimal figure dimensions
    fig_height = n_rows * 175  # 175px per row
    fig_width = n_cols * 150   # 150px per column
    
    # Convert beliefs to numpy for visualization
    nn_beliefs_np = nn_beliefs.reshape(-1, nn_beliefs.shape[-1]).detach().cpu().numpy()
    nn_probs_np = nn_probs.reshape(-1).detach().cpu().numpy()
    base_dim = nn_beliefs_np.shape[-1]
    
    # Generate colors for points
    if base_dim == 3:
        _, _, colors, _ = prepare_visualization_data(nn_beliefs, nn_probs, min_alpha=min_alpha)
    else:
        from sklearn.decomposition import PCA

        def pca_for_color(beliefs_np, n_components=3):
            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(beliefs_np)  # shape (N, 3)
            return pcs

        pcs_3d = pca_for_color(nn_beliefs_np, 3)
        min_vals = pcs_3d.min(axis=0)
        max_vals = pcs_3d.max(axis=0)
        range_vals = (max_vals - min_vals) + 1e-8

        colors = []
        alpha_vals = compute_alpha_float(nn_probs_np, scale=1, min_alpha=min_alpha)
        for i in range(len(pcs_3d)):
            r = int(255 * (pcs_3d[i,0] - min_vals[0]) / range_vals[0])
            g = int(255 * (pcs_3d[i,1] - min_vals[1]) / range_vals[1])
            b = int(255 * (pcs_3d[i,2] - min_vals[2]) / range_vals[2])
            a = alpha_vals[i]
            colors.append(f"rgba({r},{g},{b},{a:.2f})")
    
    # Weighted PCA or custom projections (processed once, used for each row)
    if vis_params["use_pca"]:
        mu, components = weighted_pca(nn_beliefs_np, nn_probs_np, n_components=pca_dims)
        nn_beliefs_pca = apply_pca_projection(nn_beliefs_np, mu, components)
        
        # If we're doing 3D PCA but want to use specific dimensions
        if pca_dims > 2 and vis_params["inds"] is not None:
            nn_beliefs_2d = nn_beliefs_pca[:, vis_params["inds"]]
        else:
            # Just use the first two dimensions
            nn_beliefs_2d = nn_beliefs_pca[:, :2]
    else:
        if vis_params["project_to_simplex"] and base_dim == 3:
            nn_beliefs_2d = _project_to_simplex(nn_beliefs.reshape(-1, 3))
        elif vis_params["inds"] is not None:
            nn_beliefs_2d = nn_beliefs_np[:, vis_params["inds"]]
        else:
            nn_beliefs_2d = nn_beliefs_np[:, :2]
    
    # Plot each layer as a row
    for row_idx, layer_name in enumerate(layers_to_plot, start=1):
        predictions_list = valid_layers[layer_name]
        
        # --- Ground Truth plot (first column of each row) ---
        if vis_params["coms"]:
            # Calculate centers of mass for ground truth
            unique_dict = defaultdict(list)
            for i, b in enumerate(nn_beliefs_np):
                key = tuple(np.round(b, 5))
                unique_dict[key].append(i)
            
            com_ground_x = []
            com_ground_y = []
            com_colors = []
            for b_key, idx_list in unique_dict.items():
                group_w = nn_probs_np[idx_list]
                coords_2d = nn_beliefs_2d[idx_list] if not vis_params["project_to_simplex"] or base_dim != 3 else nn_beliefs_2d[:, idx_list].T
                
                wsum = group_w.sum()
                if wsum > 0:
                    com_x = np.sum(coords_2d[:, 0] * group_w) / wsum if not vis_params["project_to_simplex"] or base_dim != 3 else np.sum(coords_2d[0] * group_w) / wsum
                    com_y = np.sum(coords_2d[:, 1] * group_w) / wsum if not vis_params["project_to_simplex"] or base_dim != 3 else np.sum(coords_2d[1] * group_w) / wsum
                else:
                    com_x = coords_2d[:, 0].mean() if not vis_params["project_to_simplex"] or base_dim != 3 else coords_2d[0].mean()
                    com_y = coords_2d[:, 1].mean() if not vis_params["project_to_simplex"] or base_dim != 3 else coords_2d[1].mean()
                
                com_ground_x.append(com_x)
                com_ground_y.append(com_y)
                com_colors.append(colors[idx_list[0]])
            
            gt_trace = go.Scattergl(
                x=com_ground_x,
                y=com_ground_y,
                mode='markers',
                marker=dict(
                    color=com_colors,
                    size=vis_params["gt_size"] * 1.5,
                    opacity=1
                ),
                name=f"Ground Truth (COM)"
            )
        else:
            # Regular scatter plot for ground truth
            if vis_params["project_to_simplex"] and base_dim == 3:
                x_beliefs, y_beliefs = nn_beliefs_2d[0], nn_beliefs_2d[1]
            else:
                x_beliefs, y_beliefs = nn_beliefs_2d[:, 0], nn_beliefs_2d[:, 1]
                
            gt_trace = go.Scattergl(
                x=x_beliefs,
                y=y_beliefs,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=vis_params["gt_size"]
                ),
                name="Belief States"
            )
        
        fig.add_trace(gt_trace, row=row_idx, col=1)
        
        # --- Predictions plots (subsequent columns) ---
        for col_idx, predictions in enumerate(predictions_list, start=2):
            # Process predictions for this layer and checkpoint
            preds_np = predictions.reshape(-1, predictions.shape[-1])
            
            if vis_params["use_pca"]:
                preds_pca = apply_pca_projection(preds_np, mu, components)
                
                # If we're doing 3D PCA but want to use specific dimensions
                if pca_dims > 2 and vis_params["inds"] is not None:
                    preds_2d = preds_pca[:, vis_params["inds"]]
                else:
                    # Just use the first two dimensions
                    preds_2d = preds_pca[:, :2]
            else:
                if vis_params["project_to_simplex"] and base_dim == 3:
                    preds_2d = _project_to_simplex(predictions.reshape(-1, 3))
                elif vis_params["inds"] is not None:
                    preds_2d = preds_np[:, vis_params["inds"]]
                else:
                    preds_2d = preds_np[:, :2]
            
            if vis_params["coms"]:
                # Calculate centers of mass for predictions
                com_preds_x = []
                com_preds_y = []
                group_colors = []
                
                for b_key, idx_list in unique_dict.items():
                    group_w = nn_probs_np[idx_list]
                    coords_2d = preds_2d[idx_list] if not vis_params["project_to_simplex"] or base_dim != 3 else preds_2d[:, idx_list].T
                    
                    wsum = group_w.sum()
                    if wsum > 0:
                        com_x = np.sum(coords_2d[:, 0] * group_w) / wsum if not vis_params["project_to_simplex"] or base_dim != 3 else np.sum(coords_2d[0] * group_w) / wsum
                        com_y = np.sum(coords_2d[:, 1] * group_w) / wsum if not vis_params["project_to_simplex"] or base_dim != 3 else np.sum(coords_2d[1] * group_w) / wsum
                    else:
                        com_x = coords_2d[:, 0].mean() if not vis_params["project_to_simplex"] or base_dim != 3 else coords_2d[0].mean()
                        com_y = coords_2d[:, 1].mean() if not vis_params["project_to_simplex"] or base_dim != 3 else coords_2d[1].mean()
                    
                    com_preds_x.append(com_x)
                    com_preds_y.append(com_y)
                    group_colors.append(colors[idx_list[0]])
                
                pred_trace = go.Scattergl(
                    x=com_preds_x,
                    y=com_preds_y,
                    mode='markers',
                    marker=dict(
                        color=group_colors,
                        size=vis_params["pred_size"] * 1.5,
                        opacity=1
                    ),
                    name=f"Checkpoint (COM)"
                )
            else:
                # Regular scatter plot for predictions
                if vis_params["project_to_simplex"] and base_dim == 3:
                    x_preds, y_preds = preds_2d[0], preds_2d[1]
                else:
                    x_preds, y_preds = preds_2d[:, 0], preds_2d[:, 1]
                    
                pred_trace = go.Scattergl(
                    x=x_preds,
                    y=y_preds,
                    mode='markers',
                    marker=dict(
                        color=colors,
                        size=vis_params["pred_size"]
                    ),
                    name=f"Checkpoint"
                )
            
            fig.add_trace(pred_trace, row=row_idx, col=col_idx)
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=80, r=20, t=30, b=30),
        height=fig_height,
        width=fig_width
    )
    
    # Hide all axis lines, ticks, labels
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title_text="",
                row=row, col=col
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title_text="",
                row=row, col=col
            )
    
    # Add layer name labels on the left side
    for row_idx, layer_name in enumerate(layers_to_plot, start=1):
        # Truncate long layer names
        display_name = layer_name
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."
            
        fig.add_annotation(
            x=-0.1,  # Place to the left of the plot
            y=0.5,    # Middle of the row
            text=display_name,
            showarrow=False,
            font=dict(size=11),
            xref="x domain",
            yref="y domain",
            xanchor="right",
            yanchor="middle",
            row=row_idx, col=1,
            textangle=270  # Rotate text 270 degrees
        )
    
    # Add checkpoint labels on the top row only
    title_font = dict(size=10)
    fig.add_annotation(
        x=0.5,
        y=1.05,
        text="Ground Truth",
        showarrow=False,
        font=title_font,
        xref="x domain",
        yref="y domain",
        xanchor="center",
        yanchor="bottom",
        row=1, col=1
    )
    
    for col_idx, ckpt_name in enumerate(checkpoint_names[:max_checkpoints], start=1):
        # Split multi-line checkpoint names for display
        title_html = ckpt_name.replace("\n", "<br>")
        
        fig.add_annotation(
            x=0.5,
            y=1.05,
            text=title_html,
            showarrow=False,
            font=title_font,
            xref="x domain",
            yref="y domain",
            xanchor="center",
            yanchor="bottom",
            row=1, col=col_idx+1
        )
    
    # Add sweep/run in the top-left corner
    if sweep_id and run_id:
        fig.add_annotation(
            x=0.01,
            y=0.99,
            text=f"Sweep: {sweep_id}<br>Run: {run_id}",
            showarrow=False,
            font=dict(size=9, color="gray"),
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left"
        )
    
    return fig


def analyze_layerwise_predictions(s3_loader, sweep_id, run_id, selected_checkpoints, nn_beliefs, nn_probs, checkpoint_names, 
                                 show_individual_plots=True, vis_params=None, consolidated_plot=False, layer_order=None):
    """
    Analyze predictions for each layer individually and all layers combined, 
    generating visualizations for each configuration.
    
    Args:
        s3_loader: Model loader for S3
        sweep_id: ID of the sweep
        run_id: ID of the run
        selected_checkpoints: List of checkpoint paths to analyze
        nn_beliefs: Ground truth beliefs
        nn_probs: Ground truth probabilities
        checkpoint_names: Display names for each checkpoint
        show_individual_plots: Whether to display plots for individual layers
        vis_params: Dictionary of visualization parameters
        consolidated_plot: Whether to create a single consolidated plot with all layers
        layer_order: Optional list specifying the order in which to display layers
    
    Returns:
        Dictionary mapping layer names to lists of checkpoint predictions
    """
    if vis_params is None:
        vis_params = {
            "gt_size": 2,
            "pred_size": 1.5,
            "project_to_simplex": False,
            "inds": [0, 1],
            "coms": False,
            "use_pca": False
        }
    
    # Load the first checkpoint to get the layer structure
    print(f"Loading first checkpoint to identify layer structure: {selected_checkpoints[0]}")
    first_model, run_config, nn_inputs, _, _ = load_model_and_extract_beliefs(
        s3_loader, sweep_id, run_id, selected_checkpoints[0]
    )
    
    # Extract activations from the first model to identify available layers
    act_extractor = ActivationExtractor("cpu")
    first_acts = act_extractor.extract_activations(first_model, nn_inputs, "transformer", TRANSFORMER_ACTIVATION_KEYS)
    available_layers = list(first_acts.keys())
    print(f"Found {len(available_layers)} layers: {available_layers}")
    
    # Data structure to store predictions for each layer across checkpoints
    all_layer_predictions = {layer: [] for layer in available_layers}
    all_layer_predictions["all_layers_combined"] = []
    
    # Process each checkpoint
    for checkpoint_idx, checkpoint in enumerate(selected_checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_idx+1}/{len(selected_checkpoints)}: {checkpoint}")
        
        # Load model
        model, run_config = s3_loader.load_checkpoint(sweep_id, run_id, checkpoint, device="cpu")
        
        # Get predictions for all layers
        try:
            layer_results, _ = get_all_layer_predictions(model, nn_inputs, nn_beliefs, nn_probs)
            
            # Store predictions for each layer
            for layer_key, results in layer_results.items():
                if layer_key not in all_layer_predictions:
                    all_layer_predictions[layer_key] = []
                    
                if results and 'predictions' in results:
                    all_layer_predictions[layer_key].append(results['predictions'])
                else:
                    print(f"Warning: No predictions found for layer {layer_key} in checkpoint {checkpoint}")
                    all_layer_predictions[layer_key].append(torch.zeros_like(nn_beliefs))
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint}: {e}")
            # Skip this checkpoint but continue with others
    
    # Create consolidated visualization if requested
    if consolidated_plot:
        print("\nCreating consolidated visualization for all layers")
        try:
            # Define a custom layer order that puts "pre" layers before "post" layers
            custom_layer_order = [
                'blocks.0.hook_resid_pre',  # Make sure this pre-layer appears first
                'blocks.0.hook_resid_post',
                'blocks.1.hook_resid_pre',  # Added pre-layer for block 1
                'blocks.1.hook_resid_post',
                'blocks.2.hook_resid_pre',  # Added pre-layer for block 2
                'blocks.2.hook_resid_post',
                'blocks.3.hook_resid_pre',  # Added pre-layer for block 3
                'blocks.3.hook_resid_post',
                'ln_final.hook_normalized',
                'all_layers_combined'
            ]
            
            try:
                # Make sure all inputs are numpy arrays
                print("Converting layer predictions to numpy...")
                numpy_layer_predictions = {}
                for layer_name, preds_list in all_layer_predictions.items():
                    numpy_layer_predictions[layer_name] = []
                    for pred in preds_list:
                        if isinstance(pred, torch.Tensor):
                            numpy_layer_predictions[layer_name].append(pred.detach().cpu().numpy())
                        else:
                            numpy_layer_predictions[layer_name].append(pred)

                print("Converting beliefs and probs to numpy...")
                if isinstance(nn_beliefs, torch.Tensor):
                    nn_beliefs_np = nn_beliefs.detach().cpu().numpy()
                else:
                    nn_beliefs_np = nn_beliefs
                    
                if isinstance(nn_probs, torch.Tensor):
                    nn_probs_np = nn_probs.detach().cpu().numpy()
                else:
                    nn_probs_np = nn_probs
                
                if "tom_q323uantum" in run_id or "mes23s3" in run_id:
                    print("Using datashader visualization...")
                    fig = create_consolidated_visualization_datashader(
                        nn_beliefs_np, nn_probs_np,
                        numpy_layer_predictions,
                        checkpoint_names,
                        sweep_id=sweep_id,
                        run_id=run_id,
                        vis_params=vis_params,
                        layer_order=layer_order if layer_order else custom_layer_order
                    )
                else:
                    print("Using standard visualization...")
                    fig = create_consolidated_visualization(
                        nn_beliefs_np, nn_probs_np,
                        numpy_layer_predictions,
                        checkpoint_names,
                        sweep_id=sweep_id,
                        run_id=run_id,
                        vis_params=vis_params,
                        layer_order=layer_order if layer_order else custom_layer_order
                    )
                
                print("Showing figure...")
                fig.show()
            except Exception as inner_e:
                print(f"Error during visualization preparation: {inner_e}")
                import traceback
                traceback.print_exc()
                raise inner_e  # Re-raise for outer exception handler
                
        except Exception as e:
            print(f"Error creating consolidated visualization: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
    
    # Create individual visualizations for each layer if requested
    elif show_individual_plots:
        for layer_key, predictions_list in all_layer_predictions.items():
            if not predictions_list:
                print(f"Skipping visualization for {layer_key} - no predictions available")
                continue
                
            print(f"\nCreating visualization for {layer_key}")
            try:
                if "tom_quantum" in run_id:
                    # Use the Datashader-based version
                    fig = create_visualizations_datashader(
                        nn_beliefs, nn_probs,
                        predictions_list,
                        checkpoint_names,
                        sweep_id=sweep_id,
                        run_id=run_id,
                        ground_truth_size=vis_params["gt_size"],
                        prediction_size=vis_params["pred_size"],
                        project_to_simplex=vis_params["project_to_simplex"],
                        inds=vis_params["inds"],
                        coms=vis_params["coms"],
                        use_pca=vis_params["use_pca"],
                        title_suffix=layer_key  # or whatever
                    )
                else:
                    # Original plotting
                    fig = create_visualizations(
                        nn_beliefs, nn_probs,
                        predictions_list,
                        checkpoint_names,
                        sweep_id=sweep_id,
                        run_id=run_id,
                        ground_truth_size=vis_params["gt_size"],
                        prediction_size=vis_params["pred_size"],
                        project_to_simplex=vis_params["project_to_simplex"],
                        inds=vis_params["inds"],
                        coms=vis_params["coms"],
                        use_pca=vis_params["use_pca"],
                        title_suffix=layer_key
                    )

                fig.show()

            except Exception as e:
                print(f"Error creating visualization for {layer_key}: {e}")
    
    return all_layer_predictions

if __name__ == "__main__":
    # Start timing the execution
    start_time = time.time()
    
    # Initialize S3 loader
    print("Initializing S3 loader...")
    s3_loader = S3ModelLoader(use_company_credentials=True)

    sweeps = s3_loader.list_sweeps()
    print(sweeps)

    # Set sweep and run ID
    sweep_id = "20241205175736"
    runs = s3_loader.list_runs_in_sweep(sweep_id)
    print(runs)

    # Uncomment the run_id you want to use (only one should be uncommented)
    # run_id = "run_16_L4_H4_DH16_DM64_post_quantum"
    run_id = "run_17_L4_H4_DH16_DM64_tom_quantum"
    # run_id = "run_18_L4_H4_DH16_DM64_tom_quantum"
    # run_id = "run_19_L4_H4_DH16_DM64_tom_quantum"
    # run_id = "run_20_L4_H4_DH16_DM64_tom_quantum"
    # run_id = "run_21_L4_H4_DH16_DM64_fanizza"
    # run_id = "run_22_L4_H4_DH16_DM64_rrxor"
    #run_id = "run_23_L4_H4_DH16_DM64_mess3"  # Currently using mess3 run

    # Get all checkpoints for this run
    all_checkpoints = s3_loader.list_checkpoints(sweep_id, run_id)
    print(f"Found {len(all_checkpoints)} checkpoints")

    # Extract step numbers and possibly validation loss
    checkpoint_steps = []
    step_to_idx = {}
    for i, ckpt in enumerate(all_checkpoints):
        filename = os.path.basename(ckpt)
        step = int(filename.replace(".pt", ""))
        checkpoint_steps.append(step)
        step_to_idx[step] = i

    # Attempt to load validation loss data
    print("Loading validation loss data...")
    try:
        loss_df = s3_loader.load_loss_from_run(sweep_id, run_id)
        if loss_df is None or 'num_tokens_seen' not in loss_df.columns or 'val_loss_mean' not in loss_df.columns:
            print("Warning: Loss data not available or invalid. Using evenly spaced checkpoints.")
            use_loss_spacing = False
        else:
            print("Successfully loaded validation loss data.")
            use_loss_spacing = True
            val_loss_data = loss_df[['num_tokens_seen', 'val_loss_mean']]
            step_to_loss = {}
            for step in checkpoint_steps:
                closest_idx = (val_loss_data['num_tokens_seen'] - step).abs().idxmin()
                step_to_loss[step] = val_loss_data.loc[closest_idx, 'val_loss_mean']
    except Exception as e:
        print(f"Error loading loss data: {e}")
        print("Falling back to evenly spaced checkpoints.")
        use_loss_spacing = False

    # Pick a subset of checkpoints (3 of them) if there are many
    if len(all_checkpoints) >= 3:
        selected_indices = [0]  # Always include the first checkpoint
        if use_loss_spacing:
            try:
                sorted_by_loss = sorted(checkpoint_steps, key=lambda s: step_to_loss[s])
                remaining = sorted_by_loss[1:]
                if len(remaining) >= 1:
                    # Take middle checkpoint based on loss
                    middle_idx = len(remaining) // 2
                    step = remaining[middle_idx]
                    selected_indices.append(step_to_idx[step])
                selected_indices.append(len(all_checkpoints) - 1)
            except Exception as e:
                print(f"Error during loss-based selection: {e}")
                print("Falling back to evenly spaced checkpoints.")
                selected_indices = [0]
                # Take middle and last checkpoint
                if len(all_checkpoints) > 2:
                    selected_indices.append(len(all_checkpoints) // 2)
                selected_indices.append(len(all_checkpoints) - 1)
        
        selected_indices = sorted(list(set(selected_indices)))
        selected_checkpoints = [all_checkpoints[i] for i in selected_indices]
    else:
        selected_checkpoints = all_checkpoints

    # Prepare checkpoint names
    checkpoint_names = []
    for ckpt in selected_checkpoints:
        filename = os.path.basename(ckpt)
        step = int(filename.replace(".pt", ""))

        if step >= 1_000_000:
            first_line = f"Ckpt {step / 1_000_000:.1f}M"
        else:
            first_line = f"Ckpt {step}"

        if use_loss_spacing and step in step_to_loss:
            second_line = f"Loss: {step_to_loss[step]:.4f}"
            title_text = f"{first_line}\n{second_line}"
        else:
            title_text = first_line

        checkpoint_names.append(title_text)

    print(f"Selected {len(selected_checkpoints)} checkpoints:")
    for i, (ckpt, name) in enumerate(zip(selected_checkpoints, checkpoint_names)):
        print(f"  {i+1}. {ckpt} ({name})")

    # Load first checkpoint to extract beliefs
    first_model, run_config, nn_inputs, nn_beliefs, nn_probs = load_model_and_extract_beliefs(
        s3_loader, sweep_id, run_id, selected_checkpoints[0]
    )

    # Decide visualization parameters based on run_id
    if 'mess3' in run_id:
        vis_params = {
            "gt_size": 2,
            "pred_size": 1.5,
            "project_to_simplex": True,
            "inds": None,
            "coms": False,
            "use_pca": False,
            "min_alpha": 0.01
        }
    elif 'fanizza' in run_id:
        vis_params = {
            "gt_size": 3.5,
            "pred_size": 3.0,
            "project_to_simplex": False,
            "inds": [1,2],
            "coms": False,
            "use_pca": False,
            "min_alpha": 0.2
        }
    elif 'rrxor' in run_id:
        vis_params = {
            "gt_size": 3,
            "pred_size": 3.0,
            "project_to_simplex": False,
            "inds": [1,2],  # Use dimensions 1,2 of PCA result
            "coms": True,
            "use_pca": True,
            "pca_dims": 3,  # PCA to 3 dimensions first
            "min_alpha": 0.7  # Higher opacity for scatter points
        }
    elif 'tom' in run_id:
        vis_params = {
            "gt_size": 1.0,
            "pred_size": 0.5,
            "project_to_simplex": False,
            "inds": [1,2],
            "coms": False,
            "use_pca": False,
            "min_alpha": 0.01
        }
    elif 'post_quantum' in run_id:
        vis_params = {
            "gt_size": 2.,
            "pred_size": 2.0,
            "project_to_simplex": False,
            "inds": [1,2],
            "coms": True,
            "use_pca": False,
            "min_alpha": 0.5
        }
    else:
        # fallback
        vis_params = {
            "gt_size": 2,
            "pred_size": 1.5,
            "project_to_simplex": False,
            "inds": [0,1],
            "coms": False,
            "use_pca": False
        }

    # Use layer-wise analysis
    print("\nRunning layer-wise analysis...")
    
    # Define a custom layer order that puts "pre" layers before "post" layers
    custom_layer_order = [
        'blocks.0.hook_resid_pre',  # Make sure this pre-layer appears first
        'blocks.0.hook_resid_post',
        'blocks.1.hook_resid_pre',  # Added pre-layer for block 1
        'blocks.1.hook_resid_post',
        'blocks.2.hook_resid_pre',  # Added pre-layer for block 2
        'blocks.2.hook_resid_post',
        'blocks.3.hook_resid_pre',  # Added pre-layer for block 3
        'blocks.3.hook_resid_post',
        'ln_final.hook_normalized',
        'all_layers_combined'
    ]
    
    # Set consolidated_plot=True to create a single visualization with all layers
    all_layer_predictions = analyze_layerwise_predictions(
        s3_loader, 
        sweep_id, 
        run_id, 
        selected_checkpoints, 
        nn_beliefs, 
        nn_probs, 
        checkpoint_names,
        show_individual_plots=True,  # Set to False when using consolidated plot
        vis_params=vis_params,
        consolidated_plot=False,  # Create a single plot with all layers
        layer_order=custom_layer_order  # Use our custom layer ordering
    )

    # Calculate and display total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTotal execution time: {minutes} minutes and {seconds} seconds")



# %%
