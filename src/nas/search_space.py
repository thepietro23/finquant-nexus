"""Phase 10: NAS Search Space Definition.

Defines candidate operations and search spaces for DARTS-based
architecture search on T-GAT (GNN) model.

Operations: linear, conv1d, attention, skip, none (zero)
Each MixedOp holds softmax architecture weights (alpha).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('nas_search_space')


# ============================================================
# Candidate Operations
# ============================================================

class LinearOp(nn.Module):
    """Standard linear transformation + LayerNorm + ELU."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return F.elu(self.norm(self.fc(x)))


class Conv1dOp(nn.Module):
    """1x1 convolution as feature mixer.

    For node features (n_nodes, dim), reshapes for conv then back.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (batch, in_dim)
        h = x.unsqueeze(-1)          # (batch, in_dim, 1)
        h = self.conv(h).squeeze(-1)  # (batch, out_dim)
        return F.elu(self.norm(h))


class AttentionOp(nn.Module):
    """Self-attention over nodes (single-head for lightweight)."""

    def __init__(self, in_dim, out_dim, num_heads=2):
        super().__init__()
        # Ensure in_dim divisible by num_heads
        if in_dim % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (n_nodes, dim) -> (1, n_nodes, dim) for attention
        h = x.unsqueeze(0)
        h, _ = self.attn(h, h, h)
        h = h.squeeze(0)
        return F.elu(self.norm(self.proj(h)))


class SkipOp(nn.Module):
    """Identity / skip connection. Projects if dims differ."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.proj(x)


class ZeroOp(nn.Module):
    """Zero operation — outputs zeros (removes this path)."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        return torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)


# ============================================================
# Operation Registry
# ============================================================

OPERATION_REGISTRY = {
    'linear': LinearOp,
    'conv1d': Conv1dOp,
    'attention': AttentionOp,
    'skip': SkipOp,
    'none': ZeroOp,
}


def create_operation(op_name, in_dim, out_dim):
    """Factory function to create an operation by name."""
    if op_name not in OPERATION_REGISTRY:
        raise ValueError(f"Unknown operation: {op_name}. "
                         f"Available: {list(OPERATION_REGISTRY.keys())}")
    if op_name == 'none':
        return ZeroOp(out_dim)
    elif op_name in ('skip',):
        return SkipOp(in_dim, out_dim)
    elif op_name == 'attention':
        return AttentionOp(in_dim, out_dim)
    else:
        return OPERATION_REGISTRY[op_name](in_dim, out_dim)


# ============================================================
# Mixed Operation (DARTS core)
# ============================================================

class MixedOp(nn.Module):
    """Weighted sum of all candidate operations (DARTS relaxation).

    During search: output = sum(softmax(alpha_i) * op_i(x))
    After search:  select argmax(alpha) as the final operation.
    """

    def __init__(self, in_dim, out_dim, op_names=None):
        super().__init__()

        if op_names is None:
            op_names = list(OPERATION_REGISTRY.keys())

        self.op_names = list(op_names)
        self.ops = nn.ModuleList([
            create_operation(name, in_dim, out_dim)
            for name in self.op_names
        ])

        # Architecture weights (learnable, optimized by DARTS outer loop)
        self.alpha = nn.Parameter(torch.zeros(len(self.op_names)))

    def forward(self, x):
        weights = F.softmax(self.alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    def get_selected_op(self):
        """Return name of operation with highest alpha."""
        idx = self.alpha.argmax().item()
        return self.op_names[idx]

    def get_weights(self):
        """Return softmax weights as dict."""
        weights = F.softmax(self.alpha, dim=0).detach()
        return {name: w.item() for name, w in zip(self.op_names, weights)}


# ============================================================
# Search Space Configuration
# ============================================================

class SearchSpace:
    """Defines the full NAS search space from config.

    Attributes:
        layers: candidate layer counts [2, 3, 4, 5, 6]
        dims: candidate hidden dimensions [32, 64, 128, 256]
        heads: candidate attention head counts [1, 2, 4, 8]
        operations: candidate operation names
    """

    def __init__(self, layers=None, dims=None, heads=None, operations=None):
        cfg = get_config('nas')

        self.layers = layers or cfg.get('search_space_layers', [2, 3, 4, 5, 6])
        self.dims = dims or cfg.get('search_space_dims', [32, 64, 128, 256])
        self.heads = heads or cfg.get('search_space_heads', [1, 2, 4, 8])
        self.operations = operations or list(OPERATION_REGISTRY.keys())
        self.top_k = cfg.get('top_k', 3)

        logger.info(f'SearchSpace: layers={self.layers}, dims={self.dims}, '
                    f'heads={self.heads}, ops={self.operations}')

    def get_summary(self):
        """Return search space summary dict."""
        total_combinations = len(self.layers) * len(self.dims) * len(self.operations)
        return {
            'layers': self.layers,
            'dims': self.dims,
            'heads': self.heads,
            'operations': self.operations,
            'n_ops': len(self.operations),
            'total_combinations': total_combinations,
        }
