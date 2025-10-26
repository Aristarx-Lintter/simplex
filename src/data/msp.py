import os
import joblib

import numpy as np
import torch

from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from src.analysis.plot_tools import weighted_reservoir_sample_indices
from src.data.handlers import deduplicate_data


def gt_to_msp_fixed_ctx(gt_path, n_ctx: int, max_len: int = None):
    gt = joblib.load(gt_path)
    beliefs = np.asarray(gt['beliefs']).astype(np.float32)
    probs   = np.asarray(gt['probs']).astype(np.float32)
    indices = np.asarray(gt['indices'], dtype=object)

    mask = np.array([len(t)==n_ctx for t in indices])
    beliefs, probs, indices = beliefs[mask], probs[mask], indices[mask]
    X_tokens = torch.tensor(np.stack([np.array(t, np.int64) for t in indices]), dtype=torch.long)
    Y_beliefs = torch.tensor(beliefs, dtype=torch.float32)
    W_probs   = torch.tensor(probs,   dtype=torch.float32)
    if max_len is not None:
        idx = weighted_reservoir_sample_indices(W_probs.numpy().astype(np.float32), k=max_len)
        return X_tokens[idx], Y_beliefs[idx], W_probs[idx], indices[idx]
    return X_tokens, Y_beliefs, W_probs, indices


def prepare_joblib(meta, run_dir):
    n_ctx = meta['n_ctx']; bos = meta['bos']
    os.makedirs(run_dir, exist_ok=True)
    for proc in meta['process_config']['processes']:
        run_cfg = {
            "model_config": {"n_ctx": n_ctx},
            "train_config": {"bos": bos},
            "process_config": proc,
        }
        X, beliefs, _, probs, _ = prepare_msp_data(run_cfg, run_cfg["process_config"])
        probs_d, beliefs_d, indices_d, _ = deduplicate_data(X, probs, beliefs)
        gt = {
                "probs": (probs_d.cpu().numpy() if torch.is_tensor(probs_d) else np.array(probs_d)).astype(np.float32),
                "beliefs": (beliefs_d.cpu().numpy() if torch.is_tensor(beliefs_d) else np.array(beliefs_d)).astype(np.float32),
                "indices": np.array(indices_d, dtype=object),
            }
        out = os.path.join(run_dir, f"{proc['name']}_ground_truth_data.joblib")
        joblib.dump(gt, out)
        print("Saved", out)