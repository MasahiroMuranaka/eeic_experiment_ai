import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,F] -> [B,T,N,E]
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(emb_dim))
        self.key = nn.Linear(emb_dim, emb_dim)
        self.val = nn.Linear(emb_dim, emb_dim)

    def forward(self, e: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        e: [B,T,N,E]
        m: [B,T,N] bool
        returns s: [B,T,E]
        """
        B, T, N, E = e.shape
        q = self.query.view(1, 1, 1, E)  # broadcast
        k = self.key(e)
        v = self.val(e)

        attn_logits = (q * k).sum(dim=-1)  # [B,T,N]
        attn_logits = attn_logits.masked_fill(~m, -1e9)
        attn = F.softmax(attn_logits, dim=-1)  # [B,T,N]
        s = (attn.unsqueeze(-1) * v).sum(dim=-2)  # [B,T,E]
        return s


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
        returns h: [B,out_dim] (last token)
        """
        if self.temporal == "gru":
            out, _ = self.encoder(s)     # [B,T,H]
            h = out[:, -1, :]
            return h
        else:
            x = self.posenc(s)           # [B,T,E]
            out = self.encoder(x)        # [B,T,E]
            h = out[:, -1, :]
            return h


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
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.K = int(K)
        self.emb_dim = int(emb_dim)
        self.temporal_name = temporal.lower().strip()

        self.person = PersonEncoder(in_dim, emb_dim)
        self.pool = AttentionPool(emb_dim)
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
        self.head = nn.Sequential(
            nn.Linear(self.temporal.out_dim, 256),
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
        e = self.person(X)      # [B,T,N,E]
        s = self.pool(e, M)     # [B,T,E]
        h = self.temporal(s)    # [B,D]
        logits = self.head(h)   # [B,K]
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
