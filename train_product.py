import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.training.generate_data import (
    load_config,
    generate_and_save_data,
    load_process_data,   
    get_process_string, 
)
from extractors import extract_transformer_kwargs, get_caches
from tokenizer import TokenizerMessBloch


loader_kwargs = dict(
    num_workers=max(4, os.cpu_count()//2),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)


def get_dataloaders(dataset, batch_sizes=(128, 256), amplifiers=(20, 12), split_ratio=0.9):
    split_idx = int(len(dataset) * split_ratio)
    train_idx, val_idx = dataset[:split_idx], dataset[split_idx:]

    train_ds = TensorDataset(train_idx[:, :-1], train_idx[:, 1:])
    val_ds = TensorDataset(val_idx[:, :-1], val_idx[:, 1:])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_sizes[0]*amplifiers[0],           
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_sizes[1]*amplifiers[1],
        shuffle=False,
        **loader_kwargs
    )
    return train_loader, val_loader


def get_dataset_from_caches(cfg):
    bloch_cache, mess3_cache = get_caches(cfg)
    X_bloch_all = torch.tensor(bloch_cache["transformer_inputs"], dtype=torch.long)
    X_mess3_all = torch.tensor(mess3_cache["transformer_inputs"], dtype=torch.long)
    steps = int(X_bloch_all.shape[0] / X_mess3_all.shape[0])
    X_mess3_all_new = torch.concat([X_mess3_all for _ in range(steps+1)])[:X_bloch_all.shape[0]]
    tokenizer = TokenizerMessBloch()
    return tokenizer.encode(X_mess3_all_new, X_bloch_all)


@torch.no_grad()
def val_loss_ce(model, val_loader):
    model.eval()
    tot, denom = 0.0, 0
    for X, Y in val_loader:
        X, Y = X.to(device), Y.to(device).long()
        logits = model(X)  # [B, L, V]
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1), reduction="sum")
        tot += loss.item(); denom += B*L
    return tot / denom


if __name__ == "__main__":
    amplifiers = (20, 12)
    CFG_PATH = "configs/experiment_config_transformer_mess3_bloch_hw.yaml"
    cfg = load_config(CFG_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset_from_caches(cfg)

    train_loader, val_loader = get_dataloaders(dataset, amplifiers=amplifiers)

    model = HookedTransformer(
        HookedTransformerConfig(
            n_ctx=cfg["model_config"]["n_ctx"]+1, 
            d_vocab=12,
            device=device,
            dtype=getattr(torch, cfg["model_config"]["dtype"]),
            **extract_transformer_kwargs(cfg)
        )
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg["sweep_config"]["train_config"]["learning_rate"][0]*amplifiers[0]
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=1e-6
    )

    best = float("inf")
    pbar = tqdm(range(cfg["train_config"]["n_epochs"]), desc="train")
    WARMUP_STEPS = 1000
    global_step = 0
    for g in optimizer.param_groups: g['lr'] = 0


    for ep in pbar:
        model.train()
        running_loss, steps = 0.0, 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            B, L, V = logits.shape
            loss = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1), reduction="mean")
            loss.backward(); optimizer.step()
            running_loss += loss.item(); steps += 1

        train_loss = running_loss / max(steps, 1)
        v = val_loss_ce()
        scheduler.step(v)

        if v < best:
            best = v
            torch.save(model.state_dict(), "best_model.pt")

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{v:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")



    

