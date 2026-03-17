"""Phase 10: DARTS — Differentiable Architecture Search.

Two search targets:
  1. T-GAT (GNN) — full DARTS with supernet + bilevel optimization
  2. RL Policy — grid search over network sizes (keeps SB3 intact)

Key concepts:
  - Supernet: contains all possible operations simultaneously
  - Architecture weights (alpha): softmax decides which ops are active
  - Bilevel: outer loop optimizes alpha on val, inner loop optimizes W on train
  - After search: discretize alpha → extract top-k architectures → retrain
"""

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.nas.search_space import MixedOp, SearchSpace, OPERATION_REGISTRY
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger('nas_darts')


# ============================================================
# Data structures
# ============================================================

@dataclass
class Architecture:
    """Represents a discovered architecture."""
    name: str
    ops: list                    # list of operation names per layer
    hidden_dim: int
    num_heads: int
    num_layers: int
    val_loss: float = 0.0
    params: int = 0
    description: str = ""


@dataclass
class SearchResult:
    """Result of a DARTS search run."""
    architectures: list = field(default_factory=list)   # top-k Architecture objects
    alpha_history: list = field(default_factory=list)    # alpha weights per epoch
    val_loss_history: list = field(default_factory=list)
    train_loss_history: list = field(default_factory=list)
    best_val_loss: float = float('inf')
    search_epochs: int = 0


# ============================================================
# T-GAT Supernet (DARTS)
# ============================================================

class TGATSupernet(nn.Module):
    """Supernet wrapping T-GAT with MixedOps at each layer.

    Instead of fixed GATConv layers, each layer is a MixedOp
    that blends linear/conv1d/attention/skip/none operations.
    Architecture weights (alpha) are optimized separately from model weights.
    """

    def __init__(self, n_features=21, hidden_dim=64, output_dim=64,
                 num_layers=3, num_heads=4, op_names=None):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # MixedOp layers (replace fixed GATConv)
        self.mixed_layers = nn.ModuleList()
        for i in range(num_layers):
            self.mixed_layers.append(
                MixedOp(hidden_dim, hidden_dim, op_names=op_names)
            )

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # GRU temporal encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.1)

        logger.info(f'TGATSupernet: features={n_features}, hidden={hidden_dim}, '
                    f'layers={num_layers}, ops={op_names or list(OPERATION_REGISTRY.keys())}')

    def get_arch_parameters(self):
        """Return only architecture weights (alpha) for outer optimization."""
        return [layer.alpha for layer in self.mixed_layers]

    def get_weight_parameters(self):
        """Return model weights (everything except alpha) for inner optimization."""
        arch_params = set(id(p) for p in self.get_arch_parameters())
        return [p for p in self.parameters() if id(p) not in arch_params]

    def encode_graph(self, data):
        """Process single graph through supernet layers."""
        x = F.elu(self.input_proj(data.x))

        for i, (mixed, norm) in enumerate(zip(self.mixed_layers, self.layer_norms)):
            residual = x
            x = mixed(x)           # MixedOp: weighted sum of all candidate ops
            x = norm(x)
            x = F.elu(x)
            x = self.drop(x)
            x = x + residual       # Residual connection

        return x

    def forward(self, graph_sequence):
        """Process sequence of graphs through supernet.

        Args:
            graph_sequence: list of PyG Data objects

        Returns:
            embeddings: (n_stocks, output_dim)
        """
        if not graph_sequence:
            raise ValueError("Empty graph sequence")

        spatial_embeddings = []
        for data in graph_sequence:
            h = self.encode_graph(data)
            spatial_embeddings.append(h)

        spatial_stack = torch.stack(spatial_embeddings, dim=1)
        gru_out, _ = self.gru(spatial_stack)
        temporal_out = gru_out[:, -1, :]

        return self.output_proj(temporal_out)

    def forward_single(self, data):
        """Process single graph (no temporal)."""
        h = self.encode_graph(data)
        return self.output_proj(h)

    def get_architecture(self):
        """Extract current discrete architecture from alpha weights."""
        ops = []
        for i, layer in enumerate(self.mixed_layers):
            selected = layer.get_selected_op()
            ops.append(selected)
        return ops

    def get_alpha_entropy(self):
        """Measure architecture weight convergence.

        Lower entropy = more decisive (alpha concentrated on one op).
        """
        entropies = []
        for layer in self.mixed_layers:
            probs = F.softmax(layer.alpha, dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropies.append(entropy.item())
        return np.mean(entropies)

    def get_all_alpha_weights(self):
        """Return alpha weights for all layers as dict."""
        result = {}
        for i, layer in enumerate(self.mixed_layers):
            result[f'layer_{i}'] = layer.get_weights()
        return result

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_size_mb(self):
        """Model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024


# ============================================================
# DARTS Searcher (Bilevel Optimization)
# ============================================================

class DARTSSearcher:
    """DARTS search for T-GAT architecture.

    Bilevel optimization:
      - Outer loop: optimize alpha (architecture) on validation data
      - Inner loop: optimize W (model weights) on training data

    After search, extract top-k architectures and retrain from scratch.
    """

    def __init__(self, n_features=21, hidden_dim=64, output_dim=64,
                 num_layers=3, num_heads=4, op_names=None,
                 arch_lr=3e-4, weight_lr=1e-3, device='cpu'):

        self.device = device
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Build supernet
        self.supernet = TGATSupernet(
            n_features=n_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            op_names=op_names,
        ).to(device)

        # Two optimizers: one for arch weights, one for model weights
        self.arch_optimizer = torch.optim.Adam(
            self.supernet.get_arch_parameters(), lr=arch_lr
        )
        self.weight_optimizer = torch.optim.Adam(
            self.supernet.get_weight_parameters(), lr=weight_lr
        )

        self.search_space = SearchSpace()
        self.result = SearchResult()

        logger.info(f'DARTSSearcher initialized: {self.supernet.count_parameters():,} params, '
                    f'{self.supernet.get_size_mb():.2f} MB')

    def _compute_loss(self, graphs, target):
        """Forward pass through supernet and compute MSE loss."""
        if len(graphs) == 1:
            embeddings = self.supernet.forward_single(graphs[0])
        else:
            embeddings = self.supernet(graphs)
        return F.mse_loss(embeddings, target)

    def search(self, train_graphs, val_graphs, train_target, val_target,
               epochs=None):
        """Run DARTS bilevel search.

        Args:
            train_graphs: list of PyG Data (training)
            val_graphs: list of PyG Data (validation)
            train_target: target embeddings for train (n_stocks, output_dim)
            val_target: target embeddings for val (n_stocks, output_dim)
            epochs: search epochs (default from config)

        Returns:
            SearchResult with top-k architectures + history
        """
        cfg = get_config('nas')
        epochs = epochs or cfg.get('darts_epochs', 50)

        logger.info(f'Starting DARTS search: {epochs} epochs')
        self.result = SearchResult()

        for epoch in range(epochs):
            # === Inner loop: update model weights W on training data ===
            self.supernet.train()
            self.weight_optimizer.zero_grad()
            train_loss = self._compute_loss(train_graphs, train_target)
            train_loss.backward()
            self.weight_optimizer.step()

            # === Outer loop: update architecture alpha on validation data ===
            self.supernet.train()
            self.arch_optimizer.zero_grad()
            val_loss = self._compute_loss(val_graphs, val_target)
            val_loss.backward()
            self.arch_optimizer.step()

            # Track history
            self.result.train_loss_history.append(train_loss.item())
            self.result.val_loss_history.append(val_loss.item())
            self.result.alpha_history.append(
                self.supernet.get_all_alpha_weights()
            )

            if val_loss.item() < self.result.best_val_loss:
                self.result.best_val_loss = val_loss.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                entropy = self.supernet.get_alpha_entropy()
                ops = self.supernet.get_architecture()
                logger.info(f'Epoch {epoch+1}/{epochs}: '
                            f'train_loss={train_loss.item():.4f}, '
                            f'val_loss={val_loss.item():.4f}, '
                            f'entropy={entropy:.3f}, ops={ops}')

        self.result.search_epochs = epochs
        logger.info(f'DARTS search complete. Best val_loss={self.result.best_val_loss:.4f}')

        return self.result

    def extract_top_k(self, k=None):
        """Extract top-k architectures from search result.

        Generates k architecture candidates by taking the current best
        plus variants from alpha weights.

        Args:
            k: number of architectures (default from config)

        Returns:
            list of Architecture objects
        """
        k = k or self.search_space.top_k

        architectures = []

        # Architecture 1: argmax of each layer's alpha (the "winner")
        best_ops = self.supernet.get_architecture()
        architectures.append(Architecture(
            name='darts_best',
            ops=best_ops,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            val_loss=self.result.best_val_loss,
            description=f'DARTS best: {" → ".join(best_ops)}',
        ))

        # Architecture 2+: explore second-best ops at each layer
        for variant_idx in range(1, k):
            variant_ops = list(best_ops)  # copy
            # For each additional variant, swap one layer to its 2nd-best op
            layer_idx = variant_idx % self.num_layers
            layer = self.supernet.mixed_layers[layer_idx]
            weights = F.softmax(layer.alpha, dim=0).detach()
            sorted_indices = weights.argsort(descending=True)

            # Pick 2nd best op for this layer
            if len(sorted_indices) > 1:
                second_best_idx = sorted_indices[1].item()
                variant_ops[layer_idx] = layer.op_names[second_best_idx]

            architectures.append(Architecture(
                name=f'darts_variant_{variant_idx}',
                ops=variant_ops,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                val_loss=self.result.best_val_loss,
                description=f'Variant {variant_idx}: {" → ".join(variant_ops)}',
            ))

        self.result.architectures = architectures
        logger.info(f'Extracted {len(architectures)} architectures')
        for arch in architectures:
            logger.info(f'  {arch.name}: {arch.ops}')

        return architectures

    def get_convergence_info(self):
        """Return convergence metrics for reporting."""
        if not self.result.alpha_history:
            return {'converged': False}

        initial_entropy = self.supernet.get_alpha_entropy()
        alpha_std = []
        for layer in self.supernet.mixed_layers:
            weights = F.softmax(layer.alpha, dim=0).detach()
            alpha_std.append(weights.std().item())

        return {
            'converged': True,
            'final_entropy': initial_entropy,
            'mean_alpha_std': np.mean(alpha_std),
            'best_val_loss': self.result.best_val_loss,
            'search_epochs': self.result.search_epochs,
            'final_architecture': self.supernet.get_architecture(),
        }


# ============================================================
# RL Policy Grid Search (Simple, keeps SB3 intact)
# ============================================================

@dataclass
class PolicyConfig:
    """RL policy network configuration."""
    net_arch: list
    name: str = ""
    val_sharpe: float = 0.0
    description: str = ""


# Candidate RL policy architectures
RL_POLICY_CANDIDATES = [
    PolicyConfig(net_arch=[64, 32], name='small', description='Lightweight [64, 32]'),
    PolicyConfig(net_arch=[128, 64], name='medium', description='Default [128, 64]'),
    PolicyConfig(net_arch=[256, 128], name='large', description='Wide [256, 128]'),
    PolicyConfig(net_arch=[128, 128, 64], name='deep', description='3-layer [128, 128, 64]'),
    PolicyConfig(net_arch=[64, 64], name='square', description='Square [64, 64]'),
]


def get_rl_policy_candidates():
    """Return list of RL policy architecture candidates for grid search."""
    return [copy.deepcopy(c) for c in RL_POLICY_CANDIDATES]


def rl_policy_grid_search(env_fn, candidates=None, train_steps=1000,
                          eval_episodes=3, device='cpu'):
    """Grid search over RL policy architectures.

    Uses SB3 PPO with different net_arch values.

    Args:
        env_fn: callable that returns a Gymnasium environment
        candidates: list of PolicyConfig (default: RL_POLICY_CANDIDATES)
        train_steps: steps per candidate (keep small for search)
        eval_episodes: evaluation episodes per candidate
        device: 'cpu' or 'cuda'

    Returns:
        list of PolicyConfig with val_sharpe filled in, sorted best-first
    """
    from src.rl.agent import create_ppo_agent, evaluate_agent

    if candidates is None:
        candidates = get_rl_policy_candidates()

    logger.info(f'RL policy grid search: {len(candidates)} candidates, '
                f'{train_steps} steps each')

    results = []
    for config in candidates:
        env = env_fn()
        try:
            agent = create_ppo_agent(
                env, device=device,
                policy_kwargs={
                    'net_arch': dict(
                        pi=config.net_arch, vf=config.net_arch
                    ),
                },
            )
            agent.learn(total_timesteps=train_steps)
            metrics = evaluate_agent(agent, env, n_episodes=eval_episodes)
            config.val_sharpe = metrics.get('mean_sharpe', 0.0)
            logger.info(f'  {config.name} [{config.net_arch}]: '
                        f'sharpe={config.val_sharpe:.3f}')
        except Exception as e:
            logger.warning(f'  {config.name} failed: {e}')
            config.val_sharpe = float('-inf')

        results.append(config)

    # Sort by Sharpe (best first)
    results.sort(key=lambda c: c.val_sharpe, reverse=True)
    return results


# ============================================================
# Report Generation
# ============================================================

def generate_nas_report(search_result, rl_results=None,
                        output_path='models/nas_report.pdf'):
    """Generate NAS report PDF with architecture + convergence plots.

    Args:
        search_result: SearchResult from DARTS
        rl_results: list of PolicyConfig from grid search (optional)
        output_path: where to save PDF
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        # Page 1: Training / Validation Loss Curves
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        if search_result.train_loss_history:
            axes[0].plot(search_result.train_loss_history, label='Train Loss', color='blue')
            axes[0].plot(search_result.val_loss_history, label='Val Loss', color='red')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('DARTS Search — Loss Convergence')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Alpha entropy over time
        if search_result.alpha_history:
            entropies = []
            for epoch_alpha in search_result.alpha_history:
                epoch_entropy = []
                for layer_name, weights in epoch_alpha.items():
                    probs = np.array(list(weights.values()))
                    entropy = -(probs * np.log(probs + 1e-8)).sum()
                    epoch_entropy.append(entropy)
                entropies.append(np.mean(epoch_entropy))

            axes[1].plot(entropies, label='Mean Alpha Entropy', color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Entropy')
            axes[1].set_title('Architecture Weight Convergence (lower = more decisive)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Discovered Architectures
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        text = "DARTS — Discovered Architectures\n"
        text += "=" * 50 + "\n\n"

        for arch in search_result.architectures:
            text += f"  {arch.name}:\n"
            text += f"    Operations: {' → '.join(arch.ops)}\n"
            text += f"    Hidden dim: {arch.hidden_dim}\n"
            text += f"    Layers: {arch.num_layers}\n"
            text += f"    Val loss: {arch.val_loss:.4f}\n\n"

        if rl_results:
            text += "\nRL Policy Grid Search Results\n"
            text += "=" * 50 + "\n\n"
            for config in rl_results[:5]:
                text += f"  {config.name} {config.net_arch}: "
                text += f"Sharpe={config.val_sharpe:.3f}\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Alpha Weights Heatmap (final epoch)
        if search_result.alpha_history:
            final_alpha = search_result.alpha_history[-1]
            fig, ax = plt.subplots(figsize=(10, 4))

            layer_names = list(final_alpha.keys())
            op_names = list(final_alpha[layer_names[0]].keys())
            data = np.array([
                [final_alpha[layer][op] for op in op_names]
                for layer in layer_names
            ])

            im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(len(op_names)))
            ax.set_xticklabels(op_names, rotation=45)
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names)
            ax.set_title('Final Architecture Weights (alpha)')
            plt.colorbar(im, ax=ax)

            # Annotate cells
            for i in range(len(layer_names)):
                for j in range(len(op_names)):
                    ax.text(j, i, f'{data[i,j]:.2f}',
                            ha='center', va='center', fontsize=9)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f'NAS report saved to {output_path}')
    return output_path
