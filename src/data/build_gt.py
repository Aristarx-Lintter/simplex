import joblib
import numpy as np
import torch
from tqdm import tqdm

from src.tokenizer import TokenizerMessBloch
from src.data.handlers import group_by_len


def build_joint_gt_cartesian(mess_gt_path, bloch_gt_path, mode='concat', bloch_block=4096, device='cpu'):
    """
    mess_gt_path, bloch_gt_path: joblib со структурами {'probs','beliefs','indices'}
      - indices: list/array of tuples (L,); beliefs: [N, S_m] и [N, S_b]; probs: [N]
    mode: 'concat' -> beliefs_m || beliefs_b, 'joint' -> кронекер (beliefs_m ⊗ beliefs_b)
    bloch_block: размер блока по Bloch для ограничения пика памяти
    """
    gm = joblib.load(mess_gt_path)
    gb = joblib.load(bloch_gt_path)

    tok = TokenizerMessBloch()
    pieces_probs, pieces_bel, pieces_inds = [], [], []

    gm_byL = group_by_len(gm['indices'])
    gb_byL = group_by_len(gb['indices'])

    common_lengths = [6]
    for L in common_lengths:
        mi = gm_byL[L]  # индексы Mess длины L
        bi = gb_byL[L]  # индексы Bloch длины L
        if not mi or not bi:
            continue

        # берём подмножества
        Pm = gm['probs'][mi].astype(np.float32)
        Pb = gb['probs'][bi].astype(np.float32)
        # нормализуем отдельно на всякий случай (гарантия сумм = 1 для длины L)
        Pm = Pm / Pm.sum()
        Pb = Pb / Pb.sum()

        M = len(mi); B = len(bi)
        # тензоры путей для L
        Xm = torch.tensor(np.stack([gm['indices'][k] for k in mi], axis=0), dtype=torch.long)  # [M, L]
        Xb_all = torch.tensor(np.stack([gb['indices'][k] for k in bi], axis=0), dtype=torch.long)  # [B, L]

        # beliefs (размерности определяем динамически)
        Bel_m = gm['beliefs'][mi].astype(np.float32)  # [M, S_m]
        Bel_b = gb['beliefs'][bi].astype(np.float32)  # [B, S_b]
        S_m = Bel_m.shape[1]
        S_b = Bel_b.shape[1]

        # идём по блокам Bloch, собираем части
        for start in tqdm(range(0, B, bloch_block)):
            end = min(start + bloch_block, B)
            Xb = Xb_all[start:end]                   # [Bb, L]
            Pb_blk = Pb[start:end]                  # [Bb]
            Bb = Xb.shape[0]

            # декартово произведение M×Bb
            Xm_rep = Xm.repeat_interleave(Bb, dim=0)          # [M*Bb, L]
            Xb_rep = Xb.repeat(M, 1)                           # [M*Bb, L]

            # кодируем пары
            joint_tokens = tok.encode(Xm_rep, Xb_rep).to(device)  # [M*Bb, L]

            # веса: внешнее произведение и flat
            P_joint_blk = (Pm[:, None] * Pb_blk[None, :]).reshape(-1).astype(np.float32)  # [M*Bb]

            # beliefs:
            if mode == 'joint':
                # кронекер по всем парам: [M,Bb,S_m*S_b]
                # делаем блоково, без гигантских матриц
                bel_m_rep = np.repeat(Bel_m, Bb, axis=0)        # [M*Bb, S_m]
                bel_b_rep = np.tile(Bel_b[start:end], (M, 1))   # [M*Bb, S_b]
                bel_blk = (bel_m_rep[:, :, None] * bel_b_rep[:, None, :]).reshape(-1, S_m * S_b).astype(np.float32)
            else:
                # concat: [M*Bb, S_m + S_b]
                bel_m_rep = np.repeat(Bel_m, Bb, axis=0)        # [M*Bb, S_m]
                bel_b_rep = np.tile(Bel_b[start:end], (M, 1))   # [M*Bb, S_b]
                bel_blk = np.concatenate([bel_m_rep, bel_b_rep], axis=1).astype(np.float32)

            # индексы как tuple
            inds_blk = [tuple(row.tolist()) for row in joint_tokens.cpu().numpy()]  # list of tuples

            pieces_probs.append(P_joint_blk)
            pieces_bel.append(bel_blk)
            pieces_inds.append(np.array(inds_blk, dtype=object))

    # склейка и нормировка
    probs_all = np.concatenate(pieces_probs, axis=0)
    probs_all = (probs_all / probs_all.sum()).astype(np.float32)
    beliefs_all = np.concatenate(pieces_bel, axis=0).astype(np.float32)
    indices_all = np.concatenate(pieces_inds, axis=0)

    out = {'probs': probs_all, 'beliefs': beliefs_all, 'indices': indices_all}
    return out


