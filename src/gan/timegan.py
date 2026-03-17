"""Phase 8: TimeGAN — Synthetic Financial Time Series Generation.

TimeGAN (Yoon et al., 2019) generates realistic synthetic time series
by combining autoencoder + adversarial training in a learned embedding space.

Components:
  1. Embedder: real data → latent space
  2. Recovery: latent space → data space
  3. Generator: random noise → latent sequences
  4. Discriminator: real vs fake in latent space
  5. Supervisor: latent → next-step latent (temporal dynamics)

Training phases:
  Phase 1: Autoencoder (embedder + recovery)
  Phase 2: Supervisor (temporal dynamics in latent space)
  Phase 3: Joint adversarial training (generator + discriminator)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('timegan')


class Embedder(nn.Module):
    """Maps real data to latent space."""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))


class Recovery(nn.Module):
    """Maps latent space back to data space."""

    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        o, _ = self.rnn(h)
        return self.fc(o)


class Generator(nn.Module):
    """Generates latent sequences from random noise."""

    def __init__(self, latent_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z):
        o, _ = self.rnn(z)
        return torch.sigmoid(self.fc(o))


class Discriminator(nn.Module):
    """Discriminates real vs generated latent sequences."""

    def __init__(self, latent_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        o, _ = self.rnn(h)
        return self.fc(o)


class Supervisor(nn.Module):
    """Captures temporal dynamics in latent space."""

    def __init__(self, latent_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h):
        o, _ = self.rnn(h)
        return torch.sigmoid(self.fc(o))


class TimeGAN:
    """TimeGAN wrapper for training and generation.

    Args:
        input_dim: Number of features per timestep
        seq_length: Sequence length for generated data
        hidden_dim: Hidden dimension for RNNs
        latent_dim: Latent space dimension
        num_layers: RNN layers
        device: 'cpu' or 'cuda'
    """

    def __init__(self, input_dim, seq_length=None, hidden_dim=None,
                 latent_dim=None, num_layers=None, device='cpu'):
        cfg = get_config('gan')
        self.input_dim = input_dim
        self.seq_length = seq_length or cfg.get('seq_length', 128)
        self.hidden_dim = hidden_dim or cfg.get('hidden_dim', 128)
        self.latent_dim = latent_dim or cfg.get('latent_dim', 64)
        self.num_layers = num_layers or cfg.get('num_layers', 3)
        self.device = torch.device(device)

        # Build networks
        self.embedder = Embedder(
            input_dim, self.hidden_dim, self.latent_dim, self.num_layers
        ).to(self.device)
        self.recovery = Recovery(
            self.latent_dim, self.hidden_dim, input_dim, self.num_layers
        ).to(self.device)
        self.generator = Generator(
            self.latent_dim, self.hidden_dim, self.num_layers
        ).to(self.device)
        self.discriminator = Discriminator(
            self.latent_dim, self.hidden_dim, self.num_layers
        ).to(self.device)
        self.supervisor = Supervisor(
            self.latent_dim, self.hidden_dim, max(1, self.num_layers - 1)
        ).to(self.device)

        self.trained = False

        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.embedder, self.recovery, self.generator,
                      self.discriminator, self.supervisor]
        )
        logger.info(f'TimeGAN: input_dim={input_dim}, seq={self.seq_length}, '
                    f'hidden={self.hidden_dim}, latent={self.latent_dim}, '
                    f'params={total_params:,}')

    def _prepare_data(self, data, batch_size=32):
        """Convert numpy data to DataLoader of sliding windows.

        Args:
            data: numpy array (n_timesteps, n_features) or (n_samples, seq_len, n_features)
            batch_size: Batch size
        """
        if data.ndim == 2:
            # Create sliding windows
            windows = []
            for i in range(len(data) - self.seq_length + 1):
                windows.append(data[i:i + self.seq_length])
            data_3d = np.array(windows, dtype=np.float32)
        else:
            data_3d = data.astype(np.float32)

        tensor = torch.from_numpy(data_3d).to(self.device)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, data, epochs=None, batch_size=None, lr=None,
              grad_accumulation=None):
        """Train TimeGAN on real data.

        Three training phases:
        1. Autoencoder (embedder + recovery)
        2. Supervisor (temporal dynamics)
        3. Joint (generator + discriminator + supervisor)

        Args:
            data: numpy array (n_timesteps, n_features) or (n_samples, seq, feat)
            epochs: Total epochs (split across phases)
            batch_size: Batch size
            lr: Learning rate
            grad_accumulation: Gradient accumulation steps
        """
        cfg = get_config('gan')
        epochs = epochs or cfg.get('epochs', 500)
        batch_size = batch_size or cfg.get('batch_size', 32)
        lr = lr or cfg.get('lr', 0.0005)
        grad_accumulation = grad_accumulation or cfg.get('grad_accumulation', 4)

        loader = self._prepare_data(data, batch_size)

        # Split epochs: 40% autoencoder, 20% supervisor, 40% joint
        ae_epochs = max(1, int(epochs * 0.4))
        sv_epochs = max(1, int(epochs * 0.2))
        joint_epochs = max(1, epochs - ae_epochs - sv_epochs)

        # Phase 1: Autoencoder
        logger.info(f'Phase 1: Autoencoder training ({ae_epochs} epochs)')
        self._train_autoencoder(loader, ae_epochs, lr, grad_accumulation)

        # Phase 2: Supervisor
        logger.info(f'Phase 2: Supervisor training ({sv_epochs} epochs)')
        self._train_supervisor(loader, sv_epochs, lr, grad_accumulation)

        # Phase 3: Joint
        logger.info(f'Phase 3: Joint training ({joint_epochs} epochs)')
        self._train_joint(loader, joint_epochs, lr, grad_accumulation)

        self.trained = True
        logger.info('TimeGAN training complete')

    def _train_autoencoder(self, loader, epochs, lr, grad_acc):
        optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) +
            list(self.recovery.parameters()), lr=lr
        )
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            for i, (batch,) in enumerate(loader):
                h = self.embedder(batch)
                x_hat = self.recovery(h)
                loss = F.mse_loss(x_hat, batch) / grad_acc
                loss.backward()
                if (i + 1) % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * grad_acc
            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(f'  AE epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}')

    def _train_supervisor(self, loader, epochs, lr, grad_acc):
        optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            for i, (batch,) in enumerate(loader):
                h = self.embedder(batch)
                h_hat = self.supervisor(h)
                # Predict next step in latent space
                loss = F.mse_loss(h_hat[:, :-1, :], h[:, 1:, :]) / grad_acc
                loss.backward()
                if (i + 1) % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * grad_acc

    def _train_joint(self, loader, epochs, lr, grad_acc):
        opt_g = torch.optim.Adam(
            list(self.generator.parameters()) +
            list(self.supervisor.parameters()), lr=lr
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        for epoch in range(epochs):
            g_loss_total = 0
            d_loss_total = 0

            for i, (batch,) in enumerate(loader):
                bs = batch.size(0)
                seq_len = batch.size(1)

                # Real latent
                h_real = self.embedder(batch).detach()

                # Generate fake
                z = torch.randn(bs, seq_len, self.latent_dim,
                                device=self.device)
                h_fake = self.generator(z)
                h_fake_sup = self.supervisor(h_fake)

                # --- Discriminator ---
                d_real = self.discriminator(h_real)
                d_fake = self.discriminator(h_fake_sup.detach())
                d_loss = (F.binary_cross_entropy_with_logits(
                    d_real, torch.ones_like(d_real)) +
                    F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake))) / 2

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # --- Generator ---
                h_fake2 = self.generator(z)
                h_fake_sup2 = self.supervisor(h_fake2)
                d_fake2 = self.discriminator(h_fake_sup2)

                g_adv = F.binary_cross_entropy_with_logits(
                    d_fake2, torch.ones_like(d_fake2))
                g_sup = F.mse_loss(
                    h_fake_sup2[:, :-1, :], h_fake2[:, 1:, :])
                # Moment matching
                g_moment = (torch.abs(h_real.mean(dim=0) -
                            h_fake_sup2.mean(dim=0)).mean() +
                            torch.abs(h_real.std(dim=0) -
                            h_fake_sup2.std(dim=0)).mean())

                g_loss = g_adv + 10 * g_sup + 100 * g_moment

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(f'  Joint epoch {epoch+1}/{epochs}: '
                            f'G={g_loss_total/len(loader):.4f}, '
                            f'D={d_loss_total/len(loader):.4f}')

    def generate(self, n_samples=100):
        """Generate synthetic time series.

        Args:
            n_samples: Number of sequences to generate

        Returns:
            numpy array (n_samples, seq_length, input_dim)
        """
        if not self.trained:
            raise RuntimeError("TimeGAN not trained yet. Call train() first.")

        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.seq_length, self.latent_dim,
                            device=self.device)
            h_fake = self.generator(z)
            h_sup = self.supervisor(h_fake)
            x_fake = self.recovery(h_sup)

        self.generator.train()
        self.supervisor.train()
        self.recovery.train()

        return x_fake.cpu().numpy()

    def get_stats(self):
        """Get model statistics."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            'embedder_params': count(self.embedder),
            'recovery_params': count(self.recovery),
            'generator_params': count(self.generator),
            'discriminator_params': count(self.discriminator),
            'supervisor_params': count(self.supervisor),
            'total_params': sum(count(m) for m in [
                self.embedder, self.recovery, self.generator,
                self.discriminator, self.supervisor]),
            'trained': self.trained,
        }
