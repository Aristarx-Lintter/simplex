import yaml, torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.training.generate_data import load_process_data


def extract_transformer_kwargs(cfg):
    model_cfg = cfg["sweep_config"]["model_config"]
    pure_model = cfg["model_config"]
    return {
        "n_layers": model_cfg["n_layers"][0],
        "n_heads": model_cfg["n_heads"][0],
        "d_head":  model_cfg["d_head"][0],
        "d_model": model_cfg["d_head"][0] * model_cfg["n_heads"][0],
        "d_mlp":  4 * (model_cfg["d_head"][0] * model_cfg["n_heads"][0]),
        "act_fn": pure_model["act_fn"],
        "normalization_type": pure_model["normalization_type"],
        "attn_only": pure_model["attn_only"],
        "seed": pure_model["seed"]
    }


def _single_run_cfg(base_cfg: dict, process_cfg: dict) -> dict:
    return {
        "process_config": process_cfg,
        "model_config": {"n_ctx": base_cfg["model_config"]["n_ctx"]},
        "train_config": {"bos": base_cfg["train_config"]["bos"]},
    }


def get_caches(cfg):
    proc_list = cfg["sweep_config"]["process_config"]
    data_dir = cfg['global_config']['process_dir']

    proc_bloch = next(p for p in proc_list if p["name"] == "tom_quantum")
    proc_mess3 = next(p for p in proc_list if p["name"] == "mess3")

    bloch_cache = load_process_data(_single_run_cfg(cfg, proc_bloch), data_dir)
    mess3_cache = load_process_data(_single_run_cfg(cfg, proc_mess3), data_dir)

    return bloch_cache, mess3_cache