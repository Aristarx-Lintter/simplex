import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec

from src.analysis.plot_tools import alpha_from_probs, normalize


def plot_triangle_and_sphere(
    trained, random, Y_beliefs, W_probs,
    processes: list[str] = ("Mess3", "Bloch walk"),
    modes: list[str] =("Ground truth", "Trained", "Random"),
    fontsize: int = 20
):
    beliefs = torch.as_tensor(Y_beliefs).float().cpu()
    probs   = torch.as_tensor(W_probs).float().cpu()
    probs = probs / probs.sum()
    
    preds = [
        result["combined"]['predicted_beliefs']
        .float().cpu()
        .numpy().astype(np.float32) 
        for result in [trained, random]
    ]

    A = alpha_from_probs(probs)
    
    pcas = {
        mode: PCA(n_components=4).fit(pred) 
        for mode, pred in zip(modes, [beliefs, *preds])
        }
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    for row, process in enumerate(processes):
        for col, (mode, pca) in enumerate(pcas.items()):
            ax = axes[row][col]
            dec_origin = row * 2
            
            if row == 0:
                ax.set_title(mode, fontsize=fontsize)
            
            if col == 0:
                ax.set_ylabel(process, rotation=90, labelpad=40, fontsize=fontsize)
                
                xy = pca.transform(beliefs)
                x, y = xy[:,dec_origin], xy[:,dec_origin+1]
                R, G, B = list(map(normalize, [x, y, np.sqrt(x**2 + y**2)]))
                colors = np.stack([R, G, B, A], axis=1).astype(np.float32)
            else:
                Z = pca.transform(preds[col-1]) 
                x, y = Z[:, dec_origin], Z[:, dec_origin+1]
           
            
            ax.scatter(x, y, s=2.0, c=colors, marker='.')
            del_frames(ax)
        
    plt.tight_layout()
    plt.show()
    

def plot_projections(
    result,
    beliefs, probs,
    projections: int = 4,
    fontsize: int = 20,
):
    pred = (
        result["combined"]['predicted_beliefs']
        .float().cpu().numpy().astype(np.float32) 
    )
    template = lambda i: f"({i}, {i+1})"
    labels = list(map(template, range(1, projections)))
    
    pca = PCA(n_components=projections).fit(pred)
    xy_pca = PCA(n_components=projections).fit(beliefs)
    Z = pca.transform(pred)
    
    fig = plt.figure(figsize=((projections - 1) * 5, 6), constrained_layout=True)
    gs  = GridSpec(nrows=2, ncols=projections - 1, height_ratios=[0.08, 1], figure=fig)
    for col, projection in enumerate(labels):
        hx = fig.add_subplot(gs[0, col])
        hx.text(0.5, 0.0, projection, ha='center', va='bottom', fontsize=fontsize,
            transform=hx.transAxes)
        del_frames(hx)
        
        ax = fig.add_subplot(gs[1, col])
        x, y = Z[:, col], Z[:, col+1]
        colors = get_colors(xy_pca, beliefs, probs, col)
        ax.scatter(x, y, s=2.0, c=colors, marker='.')
        del_frames(ax)
    
    plt.tight_layout()
    plt.show()


def get_colors(pca, beliefs, probs, dec_origin):
    A = alpha_from_probs(probs)
    xy = pca.transform(beliefs)
    x, y = xy[:,dec_origin], xy[:,dec_origin+1]
    R, G, B = list(map(normalize, [x, y, np.sqrt(x**2 + y**2)]))
    return np.stack([R, G, B, A], axis=1).astype(np.float32)


def del_frames(ax):
    ax.set_aspect('equal')
    ax.set_xticks([])  
    ax.set_yticks([])   
    for spine in ax.spines.values():  
        spine.set_visible(False)

def label_subplots(axes, row_labels=None, col_labels=None):
    for ax, col in zip(axes[0], col_labels):
        ax.set_title(col)
    for ax, row in zip(axes[:,0], row_labels):
        ax.set_ylabel(row, rotation=90, labelpad=40)
