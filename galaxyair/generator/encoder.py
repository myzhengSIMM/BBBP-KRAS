"""Bidirectional GRU encoder for the SMILES VAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import Tensor


class SmilesEncoder(nn.Module):
    """Bidirectional GRU encoder that maps SMILES sequences to a latent space.

    Parameters
    ----------
    vocab_size:
        Number of unique characters in the vocabulary.
    hidden_size:
        GRU hidden state dimension.
    latent_size:
        Dimension of the latent vector z.
    pad_idx:
        Padding token index (used to mask embeddings).
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout probability between GRU layers (ignored when num_layers == 1).
    device:
        Computation device.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        latent_size: int,
        pad_idx: int,
        num_layers: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_idx,
        )
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project concatenated forward+backward hidden states to latent space
        self.hidden_to_mean = nn.Linear(num_layers * 2 * hidden_size, latent_size)
        self.hidden_to_log_var = nn.Linear(num_layers * 2 * hidden_size, latent_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a batch of SMILES sequences.

        Parameters
        ----------
        inputs:
            Integer token tensor of shape (batch, max_seq_len).
        lengths:
            Sequence lengths of shape (batch,).

        Returns
        -------
        mean, log_var
            Each of shape (batch, latent_size).
        """
        batch_size = inputs.size(0)

        # Sort by descending length (required for pack_padded_sequence)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        sorted_inputs = inputs[sorted_idx]

        embedded = self.embedding(sorted_inputs)  # (batch, seq, hidden)
        packed = rnn_utils.pack_padded_sequence(
            embedded, sorted_lengths.cpu().tolist(), batch_first=True
        )
        _, hidden = self.rnn(packed)  # hidden: (num_layers*2, batch, hidden)

        # Reshape to (batch, num_layers*2*hidden)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        sorted_mean = self.hidden_to_mean(hidden)
        sorted_log_var = self.hidden_to_log_var(hidden)

        # Restore original order
        _, original_idx = torch.sort(sorted_idx, descending=False)
        mean = sorted_mean[original_idx]
        log_var = sorted_log_var[original_idx]

        return mean, log_var

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_latent(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick: z = mean + eps * std.

        Parameters
        ----------
        mean, log_var:
            Each of shape (batch, latent_size).

        Returns
        -------
        Tensor of shape (batch, latent_size).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
