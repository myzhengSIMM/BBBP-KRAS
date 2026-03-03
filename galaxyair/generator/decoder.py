"""GRU decoder for the SMILES VAE, conditioned on a latent vector."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SmilesDecoder(nn.Module):
    """Unidirectional GRU decoder conditioned on latent vector z.

    The latent vector is concatenated to the embedding at every step,
    allowing the decoder to remain conditioned throughout the generation.

    Parameters
    ----------
    vocab_size:
        Number of unique characters in the vocabulary.
    hidden_size:
        GRU hidden state dimension.
    latent_size:
        Dimension of the latent vector z.
    pad_idx:
        Padding token index.
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
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project (embedding + latent) → hidden before feeding into RNN
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_size + latent_size, hidden_size),
            nn.ReLU(),
        )
        self.output_to_logits = nn.Linear(hidden_size, vocab_size)

    # ------------------------------------------------------------------
    # Forward (teacher-forcing mode)
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: Tensor,
        latent: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Decode with teacher forcing (training) or single-step (generation).

        Parameters
        ----------
        inputs:
            Token indices.
            - Teacher-forcing: shape (batch, seq_len)
            - Single-step:     shape (batch, 1)
        latent:
            Latent vector of shape (batch, latent_size).
        hidden:
            GRU hidden state of shape (num_layers, batch, hidden_size).
            Pass None to use the teacher-forcing path.

        Returns
        -------
        Teacher-forcing:
            logits of shape (batch, seq_len, vocab_size).
        Single-step:
            (logits, new_hidden) where logits has shape (batch, 1, vocab_size).
        """
        if hidden is None:
            return self._forward_teacher_forcing(inputs, latent)
        return self._forward_single_step(inputs, latent, hidden)

    def init_hidden(self, batch_size: int) -> Tensor:
        """Return a zero-filled initial hidden state."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=self.device
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_teacher_forcing(self, inputs: Tensor, latent: Tensor) -> Tensor:
        seq_len = inputs.size(1)

        embedded = self.embedding(inputs)                          # (batch, seq, hidden)
        latent_expanded = latent.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, latent)
        projected = self.input_projection(
            torch.cat([embedded, latent_expanded], dim=2)
        )  # (batch, seq, hidden)

        out, _ = self.rnn(projected)                               # (batch, seq, hidden)
        logits = self.output_to_logits(out)                        # (batch, seq, vocab)
        return logits

    def _forward_single_step(
        self, inputs: Tensor, latent: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(inputs)                          # (batch, 1, hidden)
        projected = self.input_projection(
            torch.cat([embedded, latent.unsqueeze(1)], dim=2)
        )  # (batch, 1, hidden)

        out, new_hidden = self.rnn(projected, hidden)              # (batch, 1, hidden)
        logits = self.output_to_logits(out)                        # (batch, 1, vocab)
        return logits, new_hidden
