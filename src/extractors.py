import torch

from epsilon_transformers.training.dataloader import get_dataloader_and_loss_lower_bound_from_process, get_dataloader_from_data
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

def _iter_processes(cfg: dict) -> list[dict]:
    # Accept both the sweep config (generation time) and run config (launcher time)
    if 'sweep_config' in cfg:
        proc_list = cfg['sweep_config']['process_config']
    else:
        proc = cfg['process_config']
        proc_list = [proc] if proc is not None else []

    out = []
    for p in proc_list:
        if p.get('name') == 'mixture':
            out.extend(p.get('processes', []))
        else:
            out.append(p)
    if isinstance(cfg['process_config'], dict) and cfg['process_config']['name'] == 'mixture':
        out.extend(cfg['process_config']["processes"])
    return out


def get_caches(cfg):
    proc_list = _iter_processes(cfg)
    data_dir = cfg['global_config']['process_dir']

    proc_bloch = next(p for p in proc_list if p["name"] == "tom_quantum")
    proc_mess3 = next(p for p in proc_list if p["name"] == "mess3")

    print('proc_bloch:', proc_bloch)
    print('proc_mess3:', proc_mess3)
    print('cfg:', cfg)
    print('data_dir:', data_dir)

    bloch_cache = load_process_data(_single_run_cfg(cfg, proc_bloch), data_dir)
    mess3_cache = load_process_data(_single_run_cfg(cfg, proc_mess3), data_dir)

    return bloch_cache, mess3_cache, {'processes': [proc_bloch, proc_mess3]}



def load_data(config, device):
    # Try to load pre-generated data
    process_data = load_process_data(config, config['global_config']['process_dir'])
    print("process_data:", process_data)
    
    if process_data is not None:
        # Data was pre-generated, load it
        dataloader, d_vocab = get_dataloader_from_data(
            process_data['transformer_inputs'],
            process_data['probs'],
            config['train_config']['batches_per_epoch'],
            config['train_config']['batch_size'],
            device
        )
        loss_lower_bound = torch.from_numpy(process_data['loss_lower_bound']).to(device)
    else:
        # Data wasn't pre-generated, generate it now
        dataloader, loss_lower_bound, d_vocab = get_dataloader_and_loss_lower_bound_from_process(
            process_params=config['process_config'],
            n_ctx=config['model_config']['n_ctx'],
            bos=config['train_config']['bos'],
            batches_per_epoch=config['train_config']['batches_per_epoch'],
            batch_size=config['train_config']['batch_size'],
            device=device,
        )
    return dataloader, loss_lower_bound, d_vocab