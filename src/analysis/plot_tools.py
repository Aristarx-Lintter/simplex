import numpy as np


def alpha_from_probs(w, min_alpha=0.15, mode="cbrt"):
    w = np.clip(w / (w.max() + 1e-8), 0, 1)
    if mode == "cbrt": a = np.cbrt(w)
    elif mode == "sqrt": a = np.sqrt(w)
    elif mode == "log": a = np.log1p(w*100)/np.log1p(100)
    else: a = w
    return a*(1-min_alpha)+min_alpha

def normalize(v):
    v = np.asarray(v); mn, mx = np.nanmin(v), np.nanmax(v)
    return (v - mn)/(mx - mn + 1e-8)

def weighted_reservoir_sample_indices(probs, k, rng=None):
    rng = np.random.default_rng(rng)
    u = rng.random(len(probs))
    keys = np.log(u) / probs
    idx = np.argpartition(keys, kth=-k)[-k:]  # without full sort
    return idx
