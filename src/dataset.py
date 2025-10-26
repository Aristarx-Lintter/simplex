import json
import os
import torch
from torch.utils.data import IterableDataset

from epsilon_transformers.training.dataloader import get_dataloader_from_data
from src.extractors import get_caches
from src.tokenizer import TokenizerMessBloch


class ProductBatchLoader(IterableDataset):
    def __init__(
        self, X_mess, P_mess, X_bloch, P_bloch, tokenizer,
        batches_per_epoch: int, batch_size: int, device: str
        ):
        self.Xm = X_mess.long().cpu()     # [Nm, L]
        self.Pm = P_mess.float().cpu()    # [Nm]
        self.Xb = X_bloch.long().cpu()    # [Nb, L]
        self.Pb = P_bloch.float().cpu()   # [Nb]
        self.tokenizer = tokenizer
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.device = device
        self.tokens_per_epoch = batches_per_epoch * batch_size * (self.Xm.shape[1] - 1)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            im = torch.multinomial(self.Pm, self.batch_size, replacement=True)   # ≤ 2187
            ib = torch.multinomial(self.Pb, self.batch_size, replacement=True)   # ≤ 16384
            m = self.Xm[im]                                # [B, L], CPU
            b = self.Xb[ib]                                # [B, L], CPU
            prod = self.tokenizer.encode(m, b).to(self.device)  # [B, L]
            yield prod[:, :-1], prod[:, 1:]

    @torch.no_grad()
    def validation_data(self, num_pairs: int = 65536):
        im = torch.multinomial(self.Pm, num_pairs, replacement=True)
        ib = torch.multinomial(self.Pb, num_pairs, replacement=True)
        m = self.Xm[im]; b = self.Xb[ib]
        prod = self.tokenizer.encode(m, b).to(self.device)
        X = prod[:, :-1]; Y = prod[:, 1:]
        probs = (self.Pm[im] * self.Pb[ib]).to(self.device)
        probs = probs / probs.sum()
        return X, Y, probs

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


def make_dl(config, device):
    bloch_cache, mess3_cache, _ = get_caches(config)
    
    X_bloch_all = torch.tensor(bloch_cache['transformer_inputs'], dtype=torch.long)
    P_bloch_all = torch.tensor(bloch_cache['probs'], dtype=torch.float32)
    LLB_bloch = torch.tensor(bloch_cache['loss_lower_bound'], dtype=torch.float32)

    X_mess3_all = torch.tensor(mess3_cache['transformer_inputs'], dtype=torch.long)
    P_mess3_all = torch.tensor(mess3_cache['probs'], dtype=torch.float32)
    LLB_mess3 = torch.tensor(mess3_cache['loss_lower_bound'], dtype=torch.float32)
    dataloader = ProductBatchLoader(
        X_mess3_all, P_mess3_all, X_bloch_all, P_bloch_all,
        TokenizerMessBloch(),
        batches_per_epoch=config["train_config"]["batches_per_epoch"],
        batch_size=config["train_config"]["batch_size"],
        device=device
    )
    loss_lower_bound = (LLB_bloch + LLB_mess3).to(device)
    return dataloader, loss_lower_bound


def build_mixture_zip_dataset(config, device, concationation='cartesian'):

    print('config:', config)    
    
    bloch_cache, mess3_cache, process_cfg = get_caches(config)
    
    X_bloch_all = torch.tensor(bloch_cache['transformer_inputs'], dtype=torch.long)
    P_bloch_all = torch.tensor(bloch_cache['probs'], dtype=torch.float32)
    LLB_bloch = torch.tensor(bloch_cache['loss_lower_bound'], dtype=torch.float32)

    X_mess3_all = torch.tensor(mess3_cache['transformer_inputs'], dtype=torch.long)
    P_mess3_all = torch.tensor(mess3_cache['probs'], dtype=torch.float32)
    LLB_mess3 = torch.tensor(mess3_cache['loss_lower_bound'], dtype=torch.float32)

    
    tokenizer = TokenizerMessBloch()
    if concationation == 'zip_product':
        bloch_len = X_bloch_all.shape[0]
        mess3_len = X_mess3_all.shape[0]
        steps = bloch_len // mess3_len
        X_mess_rep = torch.cat([X_mess3_all for _ in range(steps + 1)], dim=0)[:bloch_len]
        P_mess_rep = torch.cat([P_mess3_all for _ in range(steps + 1)], dim=0)[:bloch_len]

        X_joint = tokenizer.encode(X_mess_rep, X_bloch_all)                     # shape [Nb, L], d_vocab=12
        P_joint = (P_bloch_all * P_mess_rep); P_joint = P_joint / P_joint.sum()  # normalized

        # joint LLB
        bloch_llb_len = min(LLB_bloch.shape[0], LLB_mess3.shape[0])
        LLB_joint = (LLB_bloch[:bloch_llb_len] + LLB_mess3[:bloch_llb_len]).to(device)

    else:
        Nm, _ = X_mess3_all.shape
        Nb, _ = X_bloch_all.shape

        m_rep = X_mess3_all.repeat_interleave(Nb, dim=0)   # [Nm*Nb, L]
        b_rep = X_bloch_all.repeat(Nm, 1)                  # [Nm*Nb, L]

        X_joint = tokenizer.encode(m_rep, b_rep)           # [Nm*Nb, L], d_vocab=12

        P_joint = (P_mess3_all.repeat_interleave(Nb) * P_bloch_all.repeat(Nm))  # [Nm*Nb]
        P_joint = P_joint / P_joint.sum()

        llb_len = min(LLB_bloch.shape[0], LLB_mess3.shape[0])
        LLB_joint = (LLB_bloch[:llb_len] + LLB_mess3[:llb_len]).to(device)

    metadata = {
        'process_config': {
            'name': 'mixture',
            'composition': 'zip_product',     # case "A"
            **process_cfg,
        },
        'n_ctx': config["model_config"]['n_ctx'],
        'd_vocab': tokenizer.vocab_size,      # 12
        'bos': config["train_config"]['bos'],
    }

    return X_joint, P_joint, LLB_joint, metadata, tokenizer.vocab_size

def load_mixture_data(config, device, base_dir):
    X_joint, P_joint, loss_lower_bound, mixture_meta, d_vocab = build_mixture_zip_dataset(config, device)

    if X_joint.shape[0] > 2**24:
        dataloader, loss_lower_bound = make_dl(config, device)
    else:
        dataloader, _ = get_dataloader_from_data(
            X_joint, P_joint,
            config['train_config']['batches_per_epoch'],
            config['train_config']['batch_size'],
            device,
        )
    # Save mixture metadata for reproducibility
    with open(os.path.join(base_dir, 'mixture_metadata.json'), 'w') as f:
        json.dump(mixture_meta, f, indent=2)

    return dataloader, loss_lower_bound, d_vocab
