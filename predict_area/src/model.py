import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# Social-STGCNN Components
# -------------------------------------------------------------------

class StgcnBlock(nn.Module):
    """
    Social-STGCNN Block
    Spatial Graph Conv + Temporal Conv + Residual
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Spatial Graph Conv (1x1 Conv represents the weight matrix W)
        self.gcn_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal Conv
        pad = (kernel_size - 1) // 2
        self.tcn_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: [B, C, T, N]
        res = self.residual(x)
        
        # Spatial Graph Conv: A * X * W
        # Permute for matmul: [B, C, T, N] -> [B, T, N, C]
        x_perm = x.permute(0, 2, 3, 1)
        
        # A: [B, T, N, N]
        # ax: [B, T, N, C]
        ax = torch.matmul(A, x_perm)
        
        # Back to [B, C, T, N]
        ax = ax.permute(0, 3, 1, 2)
        
        # Multiply weights
        gcn_out = self.gcn_conv(ax)
        
        # Temporal Conv
        tcn_out = self.tcn_conv(gcn_out)
        
        return self.relu(tcn_out + res)


# -------------------------------------------------------------------
# Main Model Wrapper
# -------------------------------------------------------------------

class SafetyNet(nn.Module):
    """
    Social-STGCNN implementation compatible with existing SafetyNet interface.
    """
    def __init__(
        self,
        in_dim: int,
        K: int,
        emb_dim: int = 64,  # This will likely be passed as 128 from config, which is fine
        stgcnn_layers: int = 3,
        stgcnn_kernel: int = 3,
        dropout: float = 0.3,
        **kwargs  # <--- 重要: temporal="gru" などの不要な引数をここで吸収する
    ):
        super().__init__()
        self.K = int(K)
        
        # emb_dim might come in as 128 (from old config), but STGCNN works well with 64.
        # We respect the passed emb_dim to keep dimensions consistent.
        C = emb_dim 

        # 1. Input Embedding
        # Maps raw features F to Channel dimension C
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, C),
            nn.PReLU()
        )

        # 2. ST-GCN Blocks
        self.blocks = nn.ModuleList()
        # Stack layers keeping channel size C
        for _ in range(stgcnn_layers):
            self.blocks.append(StgcnBlock(C, C, stgcnn_kernel, dropout))
            
        # 3. Output Head
        # Global pooling -> MLP -> K classes
        self.head = nn.Sequential(
            nn.Linear(C, 128),
            nn.GELU(),
            nn.Linear(128, K)
        )
        
        print(f"[SafetyNet] Initialized Social-STGCNN mode (in={in_dim}, ch={C}, layers={stgcnn_layers})")

    def compute_adjacency_matrix(self, X, M):
        """
        Dynamically compute Adjacency Matrix A based on distance.
        X: [B, T, N, F]
        M: [B, T, N]
        """
        # Assume first 2 dimensions are (x, y)
        coords = X[:, :, :, :2] 
        
        # Distance matrix
        diff = coords.unsqueeze(3) - coords.unsqueeze(2) # [B, T, N, N, 2]
        dist = torch.norm(diff, dim=-1) # [B, T, N, N]
        
        # Kernel function (Similarity)
        sigma = 2.0
        A = torch.exp(-dist / sigma)
        
        # Mask out invalid agents
        # M: [B, T, N] -> [B, T, N, N]
        M_mat = M.unsqueeze(3) * M.unsqueeze(2)
        A = A * M_mat.float()
        
        # Add self-loop (Identity)
        I = torch.eye(A.size(2), device=A.device).unsqueeze(0).unsqueeze(0)
        A = A + I
        
        # Row Normalize
        row_sum = A.sum(dim=-1, keepdim=True) + 1e-6
        A_norm = A / row_sum
        
        return A_norm

    def forward(self, X: torch.Tensor, M: torch.Tensor):
        """
        Standard interface:
        X: [B, T, N, F]
        M: [B, T, N]
        Returns: p [B, K], logits [B, K]
        """
        
        # 1. Adjacency Matrix
        A = self.compute_adjacency_matrix(X, M) # [B, T, N, N]
        
        # 2. Input Embedding
        x_emb = self.input_layer(X) # [B, T, N, C]
        
        # Permute for STGCNN: [B, T, N, C] -> [B, C, T, N]
        feat = x_emb.permute(0, 3, 1, 2)
        
        # 3. Apply Blocks
        for block in self.blocks:
            feat = block(feat, A)
            
        # feat: [B, C, T, N]
        
        # 4. Aggregation (Global Pooling)
        # Combine Time: Last frame
        feat_last = feat[:, :, -1, :] # [B, C, N]
        
        # Combine Nodes: Max Pooling (most critical agent features)
        # Apply mask to exclude padding agents
        M_last = M[:, -1, :] # [B, N]
        mask_expanded = M_last.unsqueeze(1) # [B, 1, N]
        
        # Fill invalid spots with -inf before max
        feat_last = feat_last.masked_fill(~mask_expanded, -1e9)
        
        # Max over N dimension -> [B, C]
        h = torch.max(feat_last, dim=2)[0]
        
        # 5. Prediction
        logits = self.head(h) # [B, K]
        p = torch.softmax(logits, dim=-1)
        
        return p, logits
