import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, m: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [..., N, D]
    m: [..., N] bool
    """
    w = m.to(dtype=x.dtype).unsqueeze(-1)
    num = (x * w).sum(dim=dim)
    den = w.sum(dim=dim).clamp_min(eps)
    return num / den


def _masked_softmax(logits: torch.Tensor, m: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe softmax with mask. If a row has no valid entries, returns all-zeros for that row.

    logits: [..., N]
    m:     [..., N] bool (True=valid)
    """
    # set invalid to a large negative finite value to avoid infs
    logits = logits.masked_fill(~m, -1e4)
    attn = torch.softmax(logits, dim=dim)
    # If all positions are masked, softmax becomes undefined (0/0). Fix by zeroing that row.
    has_any = m.any(dim=dim, keepdim=True)
    attn = torch.where(has_any, attn, torch.zeros_like(attn))
    # Re-normalize to make sure it sums to 1 for valid rows (and stays 0 for empty rows).
    attn = attn / (attn.sum(dim=dim, keepdim=True).clamp_min(eps))
    return attn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class PersonEncoder(nn.Module):
    """
    Per-person feature encoder.

    IMPORTANT: input X contains zero padding for missing people; the mask M must be applied
    after encoding to prevent linear biases from leaking signal from padded rows.
    """

    def __init__(self, in_dim: int, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            ResidualMLPBlock(emb_dim, hidden=max(emb_dim * 4, 256), dropout=dropout),
            ResidualMLPBlock(emb_dim, hidden=max(emb_dim * 4, 256), dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,F] -> [B,T,N,E]
        y = self.proj(x)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.blocks(y)
        return y


class AttentionPool(nn.Module):
    """
    Mask-safe attention pooling over the person dimension.

    e: [B,T,N,E]
    m: [B,T,N] bool
    returns s: [B,T,E]
    """

    def __init__(self, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.val = nn.Linear(emb_dim, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, e: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, T, N, E = e.shape
        # ensure bool mask
        m = m.to(dtype=torch.bool)
        # prevent padded rows from leaking via bias in the encoder (or any upstream ops)
        e = e * m.unsqueeze(-1).to(dtype=e.dtype)

        x = self.norm(e)
        k = self.key(x)
        v = self.val(x)

        q = self.query.view(1, 1, 1, E)  # broadcast
        attn_logits = (q * k).sum(dim=-1)  # [B,T,N]
        attn = _masked_softmax(attn_logits, m, dim=-1)  # [B,T,N]
        s = (attn.unsqueeze(-1) * v).sum(dim=-2)  # [B,T,E]
        s = self.drop(s)
        return s


class SetAttentionBlock(nn.Module):
    """
    Set transformer block over the person dimension, applied independently per time step.
    """

    def __init__(self, dim: int, nhead: int = 4, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [BT,N,E], key_padding_mask: [BT,N] True=pad
        # Important edge-case: if a whole row is padded (no valid people), MultiheadAttention would produce NaNs.
        valid = ~key_padding_mask
        has_any = valid.any(dim=1)  # [BT]
        if not bool(has_any.any()):
            return x

        out = x
        idx = has_any.nonzero(as_tuple=False).squeeze(1)
        x_sel = x.index_select(0, idx)
        kpm_sel = key_padding_mask.index_select(0, idx)

        y = self.norm1(x_sel)
        y, _ = self.attn(y, y, y, key_padding_mask=kpm_sel, need_weights=False)
        x_sel = x_sel + self.drop1(y)
        x_sel = x_sel + self.ff(self.norm2(x_sel))

        out = out.clone()
        out.index_copy_(0, idx, x_sel)
        return out


class TemporalAttentionPool(nn.Module):
    """
    Attention pooling over the time dimension.

    out: [B,T,D]
    returns h: [B,D]
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.val = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        z = self.norm(x)
        k = self.key(z)
        v = self.val(z)
        q = self.query.view(1, 1, D)
        logits = (q * k).sum(dim=-1)  # [B,T]
        # time steps are always valid in this project, but keep it safe anyway
        m = torch.ones((B, T), dtype=torch.bool, device=x.device)
        attn = _masked_softmax(logits, m, dim=-1)  # [B,T]
        h = (attn.unsqueeze(-1) * v).sum(dim=-2)  # [B,D]
        h = self.drop(h)
        return h


class SpatioTemporalTransformer(nn.Module):
    """
    Strong option: flatten (time, person) tokens -> TransformerEncoder with key_padding_mask from M.

    - CLS token is always present (unmasked), preventing NaNs when all person tokens are masked.
    - 2D learned positional embeddings: time + person-index.

    Inputs:
      e: [B,T,N,E]
      M: [B,T,N] bool (True=valid)
    Output:
      h: [B,E] (CLS)
    """

    def __init__(
        self,
        emb_dim: int,
        *,
        tf_layers: int,
        tf_nhead: int,
        tf_ff: int,
        tf_dropout: float,
        tf_norm_first: bool,
        max_T: int = 2048,
        max_N: int = 512,
    ):
        super().__init__()
        emb_dim = int(emb_dim)
        tf_nhead = int(tf_nhead)
        if emb_dim % tf_nhead != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by tf_nhead ({tf_nhead})")

        self.emb_dim = emb_dim
        self.max_T = int(max_T)
        self.max_N = int(max_N)

        self.cls = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.time_emb = nn.Embedding(self.max_T, emb_dim)
        self.person_emb = nn.Embedding(self.max_N, emb_dim)
        self.pre_norm = nn.LayerNorm(emb_dim)
        self.pre_drop = nn.Dropout(float(tf_dropout))

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=tf_nhead,
            dim_feedforward=int(tf_ff),
            dropout=float(tf_dropout),
            batch_first=True,
            activation="gelu",
            norm_first=bool(tf_norm_first),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(tf_layers))
        self.post_norm = nn.LayerNorm(emb_dim)

    def forward(self, e: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        B, T, N, E = e.shape
        if E != self.emb_dim:
            raise ValueError(f"emb_dim mismatch: got E={E}, expected {self.emb_dim}")
        if T > self.max_T:
            raise ValueError(f"T={T} exceeds max_T={self.max_T}")
        if N > self.max_N:
            raise ValueError(f"N={N} exceeds max_N={self.max_N}")

        M = M.to(dtype=torch.bool)

        x = e.reshape(B, T * N, E)              # [B,S,E]
        tok_pad = (~M).reshape(B, T * N)        # [B,S] True=pad

        t_idx = torch.arange(T, device=e.device).repeat_interleave(N)  # [S]
        n_idx = torch.arange(N, device=e.device).repeat(T)             # [S]
        pos = self.time_emb(t_idx) + self.person_emb(n_idx)            # [S,E]
        x = x + pos.unsqueeze(0).to(dtype=x.dtype)

        cls = self.cls.view(1, 1, E).expand(B, 1, E).to(dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)  # [B,1+S,E]
        tok_pad = torch.cat(
            [torch.zeros((B, 1), dtype=torch.bool, device=e.device), tok_pad],
            dim=1,
        )  # [B,1+S]

        x = self.pre_norm(x)
        x = self.pre_drop(x)
        out = self.encoder(x, src_key_padding_mask=tok_pad)  # [B,1+S,E]
        out = self.post_norm(out)
        return out[:, 0, :]


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal PE for Transformer.
    x: [B,T,E] -> x + pe[:T]
    """
    def __init__(self, emb_dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, emb_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)  # [1,T,E]


class TemporalEncoder(nn.Module):
    """
    Switchable temporal encoder: GRU or TransformerEncoder
    """
    def __init__(
        self,
        temporal: str,
        emb_dim: int,
        # GRU params
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        # Transformer params
        tf_layers: int = 2,
        tf_nhead: int = 4,
        tf_ff: int = 512,
        tf_dropout: float = 0.1,
        tf_norm_first: bool = True,
        pe_max_len: int = 2048,
    ):
        super().__init__()
        temporal = temporal.lower().strip()
        if temporal not in ("gru", "transformer"):
            raise ValueError(f"temporal must be 'gru' or 'transformer', got {temporal}")
        self.temporal = temporal

        self.dropout = float(tf_dropout)
        if temporal == "gru":
            self.encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
            )
            self.out_dim = rnn_hidden
            self.posenc = None

        else:
            # Transformer uses emb_dim as d_model
            if emb_dim % tf_nhead != 0:
                raise ValueError(f"emb_dim ({emb_dim}) must be divisible by tf_nhead ({tf_nhead})")
            self.posenc = SinusoidalPositionalEncoding(emb_dim, max_len=pe_max_len)
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=tf_nhead,
                dim_feedforward=tf_ff,
                dropout=tf_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=tf_norm_first,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=tf_layers)
            self.out_dim = emb_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [B,T,E]
        returns out: [B,T,out_dim]
        """
        if self.temporal == "gru":
            out, _ = self.encoder(s)     # [B,T,H]
            return out
        else:
            x = self.posenc(s)           # [B,T,E]
            out = self.encoder(x)        # [B,T,E]
            return out


class SafetyNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        K: int,
        emb_dim: int = 128,
        temporal: str = "gru",
        # GRU params
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        # Transformer params
        tf_layers: int = 2,
        tf_nhead: int = 4,
        tf_ff: int = 512,
        tf_dropout: float = 0.1,
        tf_norm_first: bool = True,
        # Stronger modeling knobs (optional; keep defaults for backward compatibility)
        set_layers: int = 2,
        set_nhead: int = 4,
        set_ff: int = 512,
        # Spatio-temporal embedding caps (only used for temporal="transformer")
        st_max_T: int = 2048,
        st_max_N: int = 512,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.K = int(K)
        self.emb_dim = int(emb_dim)
        self.temporal_name = temporal.lower().strip()

        self.person = PersonEncoder(in_dim, emb_dim, dropout=float(tf_dropout))

        # modes:
        # - transformer: strongest (spatio-temporal transformer over (t,n) tokens)
        # - gru: baseline
        if self.temporal_name == "transformer":
            self.mode = "st_transformer"
            self.st = SpatioTemporalTransformer(
                emb_dim=emb_dim,
                tf_layers=tf_layers,
                tf_nhead=tf_nhead,
                tf_ff=tf_ff,
                tf_dropout=tf_dropout,
                tf_norm_first=tf_norm_first,
                max_T=int(st_max_T),
                max_N=int(st_max_N),
            )
            self.out_dim = int(emb_dim)

            # keep attributes for compatibility; not used in this mode
            self.set_layers = 0
            self.set_blocks = nn.ModuleList([])
            self.pool = None
            self.temporal = None
            self.time_pool = None
        else:
            self.mode = "gru"
            # person-set modeling per frame (BT,N,E)
            self.set_layers = int(set_layers)
            self.set_blocks = nn.ModuleList(
                [
                    SetAttentionBlock(
                        dim=emb_dim,
                        nhead=int(set_nhead),
                        ff=int(set_ff),
                        dropout=float(tf_dropout),
                    )
                    for _ in range(int(set_layers))
                ]
            )

            self.pool = AttentionPool(emb_dim, dropout=float(tf_dropout))
            self.temporal = TemporalEncoder(
                temporal=self.temporal_name,
                emb_dim=emb_dim,
                rnn_hidden=rnn_hidden,
                rnn_layers=rnn_layers,
                tf_layers=tf_layers,
                tf_nhead=tf_nhead,
                tf_ff=tf_ff,
                tf_dropout=tf_dropout,
                tf_norm_first=tf_norm_first,
            )
            self.out_dim = int(self.temporal.out_dim)
            self.time_pool = TemporalAttentionPool(self.out_dim, dropout=float(tf_dropout))

        self.head = nn.Sequential(
            nn.Linear(self.out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, K),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor):
        """
        X: [B,T,N,F]
        M: [B,T,N]
        returns p: [B,K], logits: [B,K]
        """
        # ensure correct dtypes
        M = M.to(dtype=torch.bool)

        e = self.person(X)                              # [B,T,N,E]
        e = e * M.unsqueeze(-1).to(dtype=e.dtype)       # IMPORTANT: kill padded rows

        if self.mode == "st_transformer":
            h = self.st(e, M)                           # [B,E]
        else:
            # Set-attention over persons (per time step)
            if self.set_layers > 0:
                B, T, N, E = e.shape
                bt = B * T
                x = e.reshape(bt, N, E)
                pad = (~M).reshape(bt, N)               # True=pad
                for blk in self.set_blocks:
                    x = blk(x, key_padding_mask=pad)
                    # keep padded rows zeroed (safety)
                    x = x * (~pad).unsqueeze(-1).to(dtype=x.dtype)
                e = x.reshape(B, T, N, E)

            assert self.pool is not None
            assert self.temporal is not None
            assert self.time_pool is not None
            s = self.pool(e, M)                         # [B,T,E]
            out = self.temporal(s)                      # [B,T,D]
            h = self.time_pool(out)                     # [B,D]
        logits = self.head(h)                           # [B,K]
        p = torch.softmax(logits, dim=-1)
        return p, logits
    
    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Inference helper to match sklearn-like API used by inference pipeline.

        Parameters
        - X: [B,T,N,F]
        - M: [B,T,N] (bool mask)

        Returns
        - p: [B,K] class/bin probabilities
        """
        self.eval()
        p, _ = self.forward(X, M)
        return p

class SafetyNet2(SafetyNet):
    def __init__(
        self, 
        in_dim, 
        K, 
        emb_dim = 128, 
        temporal = "gru", 
        rnn_hidden = 256, 
        rnn_layers = 2, 
        tf_layers = 2, 
        tf_nhead = 4, 
        tf_ff = 512, 
        tf_dropout = 0.1, 
        tf_norm_first = True
        ):
        super().__init__(
            in_dim, 
            K, 
            emb_dim, 
            temporal, 
            rnn_hidden, 
            rnn_layers, 
            tf_layers, 
            tf_nhead, 
            tf_ff, 
            tf_dropout, 
            tf_norm_first
        )
        self.residual_layer = nn.Sequential(
            nn.Linear(K, K//2),
            nn.GELU(),
            nn.Linear(K//2, K)
        )
        
    def forward(self, X: torch.Tensor, M: torch.Tensor):
        e = self.person(X)      # [B,T,N,E]
        s = self.pool(e, M)     # [B,T,E]
        h = self.temporal(s)    # [B,D]
        h = self.head(h)   # [B,K]
        logits = h + self.residual_layer(h)
        p = torch.softmax(logits, dim=-1)
        return p, logits
