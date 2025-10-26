import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerMessBloch:
    vocab_mess: int = 3   # |Σ_mess3|
    vocab_bloch: int = 4  # |Σ_bloch|

    @property
    def vocab_size(self) -> int:
        return self.vocab_mess * self.vocab_bloch

    # ---------- scalar encode/decode ----------
    def encode_pair(self, m: int, b: int) -> int:
        self._check_scalar(m, b)
        return m * self.vocab_bloch + b

    def decode_token(self, t: int) -> tuple[int, int]:
        self._check_token_scalar(t)
        m = t // self.vocab_bloch
        b = t %  self.vocab_bloch
        return int(m), int(b)

    # ---------- tensor/batch encode/decode (PyTorch) ----------
    def encode(self, mess3_seq: torch.Tensor, bloch_seq: torch.Tensor) -> torch.Tensor:
        """
        mess3_seq, bloch_seq: shape [...], dtype int/long
        return: tokens with same shape and device
        """
        self._check_tensor_bounds(mess3_seq, bloch_seq)
        return mess3_seq * self.vocab_bloch + bloch_seq

    def decode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: shape [...], dtype int/long
        return: (mess3_seq, bloch_seq) with same shape/device
        """
        if tokens.dtype not in (torch.int32, torch.int64):
            tokens = tokens.to(torch.long)
        if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
            raise ValueError(f"tokens out of range [0, {self.vocab_size-1}]")
        m = torch.div(tokens, self.vocab_bloch, rounding_mode='floor')
        b = tokens.remainder(self.vocab_bloch)
        return m, b

    # ---------- utils ----------
    def _check_scalar(self, m: int, b: int):
        if not (0 <= m < self.vocab_mess):  raise ValueError("m out of range")
        if not (0 <= b < self.vocab_bloch): raise ValueError("b out of range")

    def _check_token_scalar(self, t: int):
        if not (0 <= t < self.vocab_size):  raise ValueError("token out of range")

    def _check_tensor_bounds(self, m: torch.Tensor, b: torch.Tensor):
        if m.dtype not in (torch.int32, torch.int64): m = m.to(torch.long)
        if b.dtype not in (torch.int32, torch.int64): b = b.to(torch.long)
        if torch.any(m < 0) or torch.any(m >= self.vocab_mess):
            raise ValueError(f"mess3 values must be in [0, {self.vocab_mess-1}]")
        if torch.any(b < 0) or torch.any(b >= self.vocab_bloch):
            raise ValueError(f"bloch values must be in [0, {self.vocab_bloch-1}]")