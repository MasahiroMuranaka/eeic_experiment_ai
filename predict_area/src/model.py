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
        # x: [B,T,N,F]
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


class SafetyNet(nn.Module):
    def __init__(self, in_dim: int, K: int, emb_dim: int = 128, rnn_hidden: int = 256, rnn_layers: int = 2):
        super().__init__()
        self.person = PersonEncoder(in_dim, emb_dim)
        self.pool = AttentionPool(emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, 256),
            nn.GELU(),
            nn.Linear(256, K),
        )

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        X: [B,T,N,F]
        M: [B,T,N]
        returns p: [B,K]
        """
        e = self.person(X)          # [B,T,N,E]
        s = self.pool(e, M)         # [B,T,E]
        out, _ = self.rnn(s)        # [B,T,H]
        h = out[:, -1, :]           # last time
        logits = self.head(h)       # [B,K]
        p = torch.softmax(logits, dim=-1)
        return p, logits
