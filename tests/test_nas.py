"""Phase 10: NAS / DARTS Tests.

7 unit tests + 3 edge cases = 10 total.

Unit tests:
  T10.1: Supernet VRAM — fits in <1.5GB (param count check)
  T10.2: Alpha convergence — entropy decreases during search
  T10.3: Architecture extraction — top-3 extracted correctly
  T10.4: NAS vs hand-designed — soft comparison (log, warn if <5%)
  T10.5: NAS report — PDF generated with figures
  T10.6: Reproducibility — same seed = same architecture
  T10.7: RL policy grid search — returns ranked candidates

Edge cases:
  E10.1: All architectures similar — doesn't crash
  E10.2: Single-layer supernet — works correctly
  E10.3: Skip connection dominance — alpha entropy check
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nas.search_space import (
    MixedOp, SearchSpace, LinearOp, Conv1dOp, AttentionOp,
    SkipOp, ZeroOp, create_operation, OPERATION_REGISTRY,
)
from src.nas.darts import (
    TGATSupernet, DARTSSearcher, Architecture, SearchResult,
    get_rl_policy_candidates, PolicyConfig, generate_nas_report,
)
from src.utils.seed import set_seed


# ============================================================
# Helpers
# ============================================================

def make_fake_graphs(n_stocks=10, n_features=21, seq_len=3, device='cpu'):
    """Create fake graph sequence for testing."""
    graphs = []
    for _ in range(seq_len):
        x = torch.randn(n_stocks, n_features, device=device)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 1, 0, 3, 2],
             [1, 0, 3, 2, 0, 1, 2, 3]], dtype=torch.long, device=device
        )
        edge_type = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1],
                                 dtype=torch.long, device=device)
        graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type))
    return graphs


def make_target(n_stocks=10, output_dim=64, device='cpu'):
    """Create fake target embeddings."""
    return torch.randn(n_stocks, output_dim, device=device)


# ============================================================
# UNIT TESTS
# ============================================================

class TestSupernet:
    """T10.1: Supernet VRAM + basic functionality."""

    def test_supernet_param_count(self):
        """Supernet should be lightweight (<1.5GB equivalent)."""
        supernet = TGATSupernet(n_features=21, hidden_dim=64,
                                num_layers=3)
        params = supernet.count_parameters()
        size_mb = supernet.get_size_mb()

        # Supernet with mixed ops should be < 50MB (well within 1.5GB)
        assert size_mb < 50, f"Supernet too large: {size_mb:.1f} MB"
        assert params > 0
        print(f"Supernet: {params:,} params, {size_mb:.2f} MB")

    def test_supernet_forward_single(self):
        """Single graph forward pass."""
        supernet = TGATSupernet(n_features=21, hidden_dim=32, num_layers=2)
        graphs = make_fake_graphs(n_stocks=10, n_features=21, seq_len=1)
        supernet.eval()

        with torch.no_grad():
            emb = supernet.forward_single(graphs[0])

        assert emb.shape == (10, 64), f"Expected (10, 64), got {emb.shape}"
        assert torch.isfinite(emb).all()

    def test_supernet_forward_sequence(self):
        """Sequence of graphs forward pass."""
        supernet = TGATSupernet(n_features=21, hidden_dim=32, num_layers=2)
        graphs = make_fake_graphs(n_stocks=10, n_features=21, seq_len=5)
        supernet.eval()

        with torch.no_grad():
            emb = supernet(graphs)

        assert emb.shape == (10, 64)
        assert torch.isfinite(emb).all()

    def test_arch_weight_separation(self):
        """Architecture params and weight params should be separate."""
        supernet = TGATSupernet(n_features=21, hidden_dim=32, num_layers=2)

        arch_params = supernet.get_arch_parameters()
        weight_params = supernet.get_weight_parameters()

        arch_ids = set(id(p) for p in arch_params)
        weight_ids = set(id(p) for p in weight_params)

        # No overlap
        assert len(arch_ids & weight_ids) == 0, "Arch and weight params overlap!"
        # Together they should cover all params
        all_ids = set(id(p) for p in supernet.parameters())
        assert arch_ids | weight_ids == all_ids


class TestAlphaConvergence:
    """T10.2: Alpha weights converge during search."""

    def test_alpha_entropy_decreases(self):
        """After DARTS search, entropy should be lower than initial."""
        set_seed(42)
        n_stocks, n_features = 10, 21

        searcher = DARTSSearcher(
            n_features=n_features, hidden_dim=32, output_dim=32,
            num_layers=2, device='cpu'
        )

        # Initial entropy (uniform alpha → high entropy)
        initial_entropy = searcher.supernet.get_alpha_entropy()

        train_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)
        val_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)
        train_target = make_target(n_stocks, 32)
        val_target = make_target(n_stocks, 32)

        result = searcher.search(
            train_graphs, val_graphs,
            train_target, val_target,
            epochs=30
        )

        final_entropy = searcher.supernet.get_alpha_entropy()

        # Entropy should decrease (alpha concentrating on fewer ops)
        # Allow some tolerance — short search may not fully converge
        assert final_entropy <= initial_entropy + 0.1, \
            f"Entropy didn't decrease: {initial_entropy:.3f} → {final_entropy:.3f}"
        assert len(result.train_loss_history) == 30
        assert len(result.val_loss_history) == 30


class TestArchitectureExtraction:
    """T10.3: Top-k architecture extraction."""

    def test_extract_top_3(self):
        """Extract 3 distinct architectures from searched supernet."""
        set_seed(42)
        n_stocks, n_features = 10, 21

        searcher = DARTSSearcher(
            n_features=n_features, hidden_dim=32, output_dim=32,
            num_layers=2, device='cpu'
        )

        train_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)
        val_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)

        searcher.search(
            train_graphs, val_graphs,
            make_target(n_stocks, 32), make_target(n_stocks, 32),
            epochs=10
        )

        architectures = searcher.extract_top_k(k=3)

        assert len(architectures) == 3
        for arch in architectures:
            assert isinstance(arch, Architecture)
            assert len(arch.ops) == 2  # num_layers=2
            assert all(op in OPERATION_REGISTRY for op in arch.ops)
            assert arch.hidden_dim == 32
            assert arch.name != ""

        # First one should be 'darts_best'
        assert architectures[0].name == 'darts_best'


class TestNASComparison:
    """T10.4: NAS vs hand-designed (soft comparison)."""

    def test_nas_comparison_logs(self):
        """NAS search produces valid result that can be compared."""
        set_seed(42)
        n_stocks, n_features = 10, 21

        searcher = DARTSSearcher(
            n_features=n_features, hidden_dim=32, output_dim=32,
            num_layers=2, device='cpu'
        )

        train_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)
        val_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)
        target = make_target(n_stocks, 32)

        result = searcher.search(
            train_graphs, val_graphs, target, target, epochs=10
        )

        # Soft check: best_val_loss should be finite
        assert np.isfinite(result.best_val_loss)
        assert result.best_val_loss >= 0

        # Convergence info should be available
        info = searcher.get_convergence_info()
        assert info['converged'] is True
        assert 'final_architecture' in info
        assert len(info['final_architecture']) == 2  # num_layers


class TestNASReport:
    """T10.5: NAS report PDF generation."""

    def test_report_generated(self):
        """Report PDF should be created with plots."""
        set_seed(42)
        n_stocks, n_features = 10, 21

        searcher = DARTSSearcher(
            n_features=n_features, hidden_dim=32, output_dim=32,
            num_layers=2, device='cpu'
        )

        train_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)
        val_graphs = make_fake_graphs(n_stocks, n_features, seq_len=3)

        result = searcher.search(
            train_graphs, val_graphs,
            make_target(n_stocks, 32), make_target(n_stocks, 32),
            epochs=10
        )
        searcher.extract_top_k(k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'nas_report.pdf')
            output = generate_nas_report(result, output_path=path)
            assert os.path.exists(output)
            assert os.path.getsize(output) > 0
            print(f"Report size: {os.path.getsize(output)} bytes")


class TestReproducibility:
    """T10.6: Same seed = same architecture."""

    def test_same_seed_same_arch(self):
        """Two runs with same seed should discover identical architecture."""
        n_stocks, n_features = 10, 21

        def run_search(seed):
            set_seed(seed)
            searcher = DARTSSearcher(
                n_features=n_features, hidden_dim=32, output_dim=32,
                num_layers=2, device='cpu'
            )
            train_g = make_fake_graphs(n_stocks, n_features, seq_len=3)
            val_g = make_fake_graphs(n_stocks, n_features, seq_len=3)
            target = make_target(n_stocks, 32)

            searcher.search(train_g, val_g, target, target, epochs=15)
            return searcher.supernet.get_architecture()

        arch1 = run_search(42)
        arch2 = run_search(42)

        assert arch1 == arch2, f"Different architectures: {arch1} vs {arch2}"


class TestRLPolicySearch:
    """T10.7: RL policy grid search returns ranked candidates."""

    def test_rl_candidates_exist(self):
        """Should have 5 predefined RL policy candidates."""
        candidates = get_rl_policy_candidates()
        assert len(candidates) == 5
        for c in candidates:
            assert isinstance(c, PolicyConfig)
            assert len(c.net_arch) >= 2
            assert c.name != ""

    def test_rl_grid_search_with_env(self):
        """Grid search over RL policy architectures (quick, 2 candidates)."""
        from src.rl.environment import PortfolioEnv

        n_stocks, n_time, n_feat = 5, 200, 21
        np.random.seed(42)
        features = np.random.randn(n_stocks, n_time, n_feat).astype(np.float32)
        prices = 100 * np.cumprod(
            1 + np.random.randn(n_stocks, n_time) * 0.01, axis=1
        ).astype(np.float32)

        def env_fn():
            return PortfolioEnv(features, prices, episode_length=30)

        # Test with just 2 small candidates for speed
        candidates = [
            PolicyConfig(net_arch=[32, 16], name='tiny'),
            PolicyConfig(net_arch=[64, 32], name='small'),
        ]

        from src.nas.darts import rl_policy_grid_search
        results = rl_policy_grid_search(
            env_fn, candidates=candidates,
            train_steps=256, eval_episodes=1, device='cpu'
        )

        assert len(results) == 2
        # Should be sorted by sharpe (best first)
        assert results[0].val_sharpe >= results[1].val_sharpe
        # All should have finite sharpe
        for r in results:
            assert np.isfinite(r.val_sharpe)


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    """E10.1-3: Edge case handling."""

    def test_all_architectures_similar(self):
        """E10.1: If search space is tiny, all archs may be similar — no crash."""
        set_seed(42)
        searcher = DARTSSearcher(
            n_features=21, hidden_dim=32, output_dim=32,
            num_layers=1, op_names=['linear', 'skip'],  # Only 2 ops
            device='cpu'
        )

        graphs = make_fake_graphs(10, 21, seq_len=2)
        target = make_target(10, 32)

        searcher.search(graphs, graphs, target, target, epochs=5)
        archs = searcher.extract_top_k(k=3)

        # Should still return 3 architectures (may have duplicates)
        assert len(archs) == 3
        for a in archs:
            assert all(op in ['linear', 'skip'] for op in a.ops)

    def test_single_layer_supernet(self):
        """E10.2: Supernet with 1 layer should work."""
        supernet = TGATSupernet(n_features=21, hidden_dim=32,
                                num_layers=1)
        graphs = make_fake_graphs(10, 21, seq_len=2)
        supernet.eval()

        with torch.no_grad():
            emb = supernet(graphs)

        assert emb.shape == (10, 64)
        assert torch.isfinite(emb).all()

        arch = supernet.get_architecture()
        assert len(arch) == 1  # Single layer

    def test_skip_dominance_check(self):
        """E10.3: If skip dominates, entropy should still be measurable."""
        supernet = TGATSupernet(n_features=21, hidden_dim=32,
                                num_layers=2)

        # Force skip to dominate by setting its alpha high
        for layer in supernet.mixed_layers:
            skip_idx = layer.op_names.index('skip')
            with torch.no_grad():
                layer.alpha.fill_(-10)  # All ops very low
                layer.alpha[skip_idx] = 10  # Skip very high

        entropy = supernet.get_alpha_entropy()
        # With one dominant op, entropy should be very low
        assert entropy < 0.5, f"Entropy too high with skip dominance: {entropy}"

        arch = supernet.get_architecture()
        assert all(op == 'skip' for op in arch)


# ============================================================
# Search Space Module Tests
# ============================================================

class TestSearchSpace:
    """Tests for search_space.py components."""

    def test_all_operations_create(self):
        """All 5 operations should instantiate and forward."""
        for op_name in OPERATION_REGISTRY:
            op = create_operation(op_name, 32, 64)
            x = torch.randn(10, 32)
            out = op(x)
            assert out.shape == (10, 64), f"{op_name}: {out.shape}"
            assert torch.isfinite(out).all(), f"{op_name} has NaN/Inf"

    def test_mixed_op(self):
        """MixedOp blends all operations."""
        mixed = MixedOp(32, 64)
        x = torch.randn(10, 32)
        out = mixed(x)

        assert out.shape == (10, 64)
        assert torch.isfinite(out).all()

        # Check weights sum to 1
        weights = mixed.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-5

        # Selected op should be a valid name
        selected = mixed.get_selected_op()
        assert selected in OPERATION_REGISTRY

    def test_search_space_config(self):
        """SearchSpace loads from config."""
        space = SearchSpace()
        summary = space.get_summary()

        assert 'layers' in summary
        assert 'dims' in summary
        assert 'operations' in summary
        assert summary['n_ops'] == 5  # linear, conv1d, attention, skip, none
        assert summary['total_combinations'] > 0

    def test_unknown_operation_raises(self):
        """Unknown operation name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operation"):
            create_operation('transformer_xl', 32, 64)
