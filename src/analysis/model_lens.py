import os
import json
import torch
from pathlib import Path
from transformer_lens import HookedTransformer, HookedTransformerConfig

from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.run_regression_analysis import (
    _combine_layer_activations
)


def load_model(run_dir, ckpt='first and last', device='cpu'):
    with open(os.path.join(run_dir, 'hooked_model_config.json')) as f:
        d = json.load(f)
    # fix dtype: "torch.float32" -> torch.float32
    if isinstance(d.get('dtype'), str):
        d['dtype'] = getattr(torch, d['dtype'].split('.')[-1])
    cfg = HookedTransformerConfig.from_dict(d)

    if ckpt == 'first and last':
        pts = sorted(Path(run_dir).glob('*.pt'), key=lambda p: p.stat().st_mtime) 
        another = sorted(Path(run_dir).glob('*.pt')) 
        assert pts, "no checkpoints found"
        ckpt_paths = [pts[0], pts[-1], another[-1]]
    else:
        ckpt_paths = [os.path.join(run_dir, f'{ckpt}.pt')]

    models = []
    for ckpt_path in ckpt_paths:
        model = HookedTransformer(cfg, move_to_device=False).to(device)
        print(ckpt_path)
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        models.append(model)
    return models, ckpt_paths

def get_acts(model, X_tokens, device):
    actx = ActivationExtractor(device=device)
    acts = actx.extract_activations(model, X_tokens, model_type='transformer', relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS)
    nn_acts = {k: v for k, v in acts.items()}
    nn_acts['combined'] = _combine_layer_activations(nn_acts)  # обычно [N,L,D]

    return nn_acts
