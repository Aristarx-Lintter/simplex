import gc
import math
import torch
from tqdm import tqdm

from src.analysis.model_lens import get_acts


@torch.no_grad()
def run_activation_to_beliefs_regression_single(activations, beliefs, probs, rcond_values=(1e-6, 1e-5, 1e-4),
                                                batch_size=131072, device='cuda'):
    """
    One-pass weighted regression without KFold.
    - activations: [N, D] or [N, L, D] (then take the last token)
    - beliefs:     [N, S] or [N, L, S] (then take the last token when 3D)
    - probs:       [N]    (weights of examples)
    Returns a dict compatible with the analysis code: 
      {'predictions': Yhat, 'final_metrics': {...}, 'rcond': best_rcond, 'beta': beta}
    """
    X = activations
    Y = beliefs
    if X.dim() == 3:
        X = X[:, -1, :]
    if Y.dim() == 3:
        Y = Y[:, -1, :]
    X = X.contiguous().float()
    Y = Y.contiguous().float()
    w = probs.contiguous().float()

    N, D = X.shape
    S = Y.shape[1]
    dev = torch.device(device)

    # 2) Stream X' = [1, X]: accumulate XtX and XtY
    XtX = torch.zeros(D+1, D+1, device=dev, dtype=torch.float32)
    XtY = torch.zeros(D+1, S,   device=dev, dtype=torch.float32)
    for s in range(0, N, batch_size):
        e = min(s+batch_size, N)
        Xc = X[s:e].to(dev, non_blocking=True)
        Yc = Y[s:e].to(dev, non_blocking=True)
        wc = w[s:e].to(dev, non_blocking=True).clamp_min_(0)
        Wsqrt = wc.sqrt().unsqueeze(1)                         # [B,1]
        Xc_bias = torch.cat([torch.ones(Xc.size(0), 1, device=dev), Xc], dim=1)  # [B, D+1]
        Xw = Xc_bias * Wsqrt
        Yw = Yc * Wsqrt
        XtX += Xw.T @ Xw
        XtY += Xw.T @ Yw

    # 3) SVD of XtX; solve for multiple rcond, select the best by weighted MSE (on the entire set)
    U, Svals, Vh = torch.linalg.svd(XtX, full_matrices=False)
    best = None
    best_obj = math.inf
    best_pred = None

    # prepare predictions in the stream (without storing X entirely in memory)
    def predict_stream(beta):
        Yhat = torch.empty(N, S, dtype=torch.float32, device=dev)
        for s in range(0, N, batch_size):
            e = min(s+batch_size, N)
            Xc = X[s:e].to(dev, non_blocking=True)
            Xc_bias = torch.cat([torch.ones(Xc.size(0), 1, device=dev), Xc], dim=1)
            Yhat[s:e] = Xc_bias @ beta
        return Yhat

    # precompute weighted base values for metrics
    w_sum = w.sum().to(dev)
    y_mean = (Y.to(dev) * w.to(dev).unsqueeze(1)).sum(dim=0) / w_sum
    var_denom = ((Y.to(dev) - y_mean.unsqueeze(0)).pow(2).sum(dim=1) * w.to(dev)).sum() / w_sum

    for rcond in rcond_values:
        thr = rcond * Svals.max()
        Sinv = torch.where(Svals > thr, 1.0 / Svals, torch.zeros_like(Svals))
        pinv_XtX = (Vh.transpose(0,1) * Sinv) @ U.transpose(0,1)
        beta = pinv_XtX @ XtY  # [D+1, S]

        Yhat = predict_stream(beta)
        diff = Yhat - Y.to(dev)
        mse = ((diff.pow(2).sum(dim=1) * w.to(dev))).sum() / w_sum
        if mse.item() < best_obj:
            best_obj = mse.item()
            best = (rcond, beta)
            best_pred = Yhat

    rcond_best, beta_best = best
    Yhat = best_pred

    # 4) Метрики (взвешенные)
    diff = Yhat - Y.to(dev)
    mse = ((diff.pow(2).sum(dim=1) * w.to(dev))).sum() / w_sum
    rmse = mse.sqrt()
    mae = ((diff.abs().sum(dim=1) * w.to(dev))).sum() / w_sum
    dist = ((diff.norm(dim=1) * w.to(dev))).sum() / w_sum
    r2 = 1.0 - (mse / var_denom.clamp_min(1e-12))

    return {
        'predictions': Yhat,  # [N, S] (device=dev)
        'final_metrics': {
            'rmse': rmse.item(),
            'mae':  mae.item(),
            'r2':   r2.item(),
            'dist': dist.item(),
            'mse':  mse.item(),
        },
        'rcond': rcond_best,
        'beta': beta_best,     # [D+1, S]
    }


def regression(X, Y, W, device):
    res = run_activation_to_beliefs_regression_single(
        X, Y, W,
        device=device
    )
    torch.cuda.empty_cache()
    gc.collect()
    return  {
        'predicted_beliefs': res['predictions'],
        'rmse':  res['final_metrics']['rmse'],
        'mae':   res['final_metrics']['mae'],
        'r2':    res['final_metrics']['r2'],
        'dist':  res['final_metrics']['dist'],
        'mse':   res['final_metrics']['mse'],
        'cum_var_exp': None, 'cum_var_exp_zscore': None,
        'val_loss_mean': float('nan'),
    }


def run_full_regression_last_token(model, X_tokens, Y_beliefs, W_probs, pos=2, device='cpu'):
    # Last token: X_last:[N,D], Y_last:[N,S], W:[N]
    Yb = torch.as_tensor(Y_beliefs, dtype=torch.float32, device=device)
    W = torch.as_tensor(W_probs, dtype=torch.float32, device=device)  # [N]

    nn_acts = get_acts(model, X_tokens, device)
    save_data = {}
    
    for layer, X in tqdm(nn_acts.items()):
        print(layer)
        if ("post" in layer) or ('combined' in layer):
            save_data[layer] = regression(X[:, pos, :].to(device), Yb, W, device=device)
            print("RMSE:", save_data[layer]["rmse"])
        
    return save_data