"""
SMILES Variational Autoencoder with contrastive pretraining and RL fine-tuning.

Architecture
------------
- Encoder: bidirectional GRU → latent mean + log-variance
- Decoder: unidirectional GRU conditioned on latent vector z

Pretraining (Section 2.2 of paper)
------------------------------------
Contrastive learning on (Lead, Opt, Control) triplets from ChEMBL KRAS dataset:
  - Contractive loss  (Fréchet / Wasserstein-2): pulls Lead and Opt together
  - Margin loss:      pushes Control away from both Lead and Opt
  - Reconstruction loss on all three SMILES

Fine-tuning (Section 2.3 of paper)
------------------------------------
Policy-gradient RL with composite reward:
  reward = 0.4 × affinity + 0.4 × BBBp + 0.2 × QED
  (subject to Tanimoto similarity > 0.4 between source and generated molecule)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from galaxyair.generator.dataset import TripletSmilesDataset, ValidationSmilesDataset
from galaxyair.generator.decoder import SmilesDecoder
from galaxyair.generator.encoder import SmilesEncoder
from galaxyair.generator.reward import AnnealingScheduler, ReplayBuffer, RewardFunction
from galaxyair.metrics.evaluation import evaluate_metric_validation


class SmilesAutoencoder(nn.Module):
    """SMILES VAE combining contrastive pretraining and policy-gradient fine-tuning.

    Parameters
    ----------
    vocab_size:
        Vocabulary size (number of unique characters).
    hidden_size:
        GRU hidden state dimension.
    latent_size:
        Latent vector dimension.
    sos_idx, eos_idx, pad_idx:
        Special token indices from the vocabulary.
    num_layers:
        Number of GRU layers (encoder and decoder share the same depth).
    dropout:
        Dropout rate between GRU layers.
    device:
        Computation device.
    config_path:
        Optional path to a saved config CSV; when provided, architecture
        parameters are loaded from the file (used during fine-tuning).
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        latent_size: Optional[int] = None,
        sos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
        pad_idx: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Allow deferred initialization when loading from a config file
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.dropout = dropout

        if config_path is not None:
            self._load_config(config_path)

        self.encoder = SmilesEncoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            pad_idx=self.pad_idx,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device,
        )
        self.decoder = SmilesDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            pad_idx=self.pad_idx,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device,
        )
        self.to(self.device)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def pretrain(
        self,
        dataset: TripletSmilesDataset,
        *,
        batch_size: int = 100,
        total_steps: int = 50000,
        learning_rate: float = 1e-3,
        use_contractive: bool = True,
        use_margin: bool = True,
        validation_dataset: Optional[ValidationSmilesDataset] = None,
        validation_repetition_size: int = 20,
        checkpoint_step: int = 5000,
        checkpoint_path: Optional[str | Path] = None,
        display_step: int = 500,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Contrastive pretraining on (Lead, Opt, Control) triplets.

        Returns
        -------
        df_history:
            Per-step loss components.
        df_history_valid:
            Per-checkpoint validation metrics.
        """
        calc_beta = AnnealingScheduler(total_steps) if use_contractive else lambda _: 0.0
        calc_gamma = AnnealingScheduler(total_steps) if use_margin else lambda _: 0.0

        use_cuda = torch.cuda.is_available()
        step = 0
        history: List[Tuple] = []
        history_valid: List[pd.DataFrame] = []

        while step < total_steps:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                drop_last=True, pin_memory=use_cuda,
            )
            for batch in loader:
                enc_lead = dataset.encode(batch["smiles_s"], int(batch["length_s"].max()))
                len_lead = batch["length_s"]
                enc_opt = dataset.encode(batch["smiles_t"], int(batch["length_t"].max()))
                len_opt = batch["length_t"]
                enc_con = dataset.encode(batch["smiles_n"], int(batch["length_n"].max()))
                len_con = batch["length_n"]

                beta = calc_beta(step)
                gamma = calc_gamma(step)
                losses = self._pretrain_step(
                    enc_lead, len_lead,
                    enc_opt, len_opt,
                    enc_con, len_con,
                    lr=learning_rate, beta=beta, gamma=gamma,
                )
                history.append((*losses, beta, gamma))

                if checkpoint_path is not None and step % checkpoint_step == 0:
                    self.save_weights(checkpoint_path)

                if step % display_step == 0:
                    log = (
                        f"[{step:08d}/{total_steps:08d}]"
                        f"  loss: {losses[0]:.4f}"
                        f"  recon_lead: {losses[1]:.4f}"
                        f"  recon_opt: {losses[2]:.4f}"
                        f"  recon_con: {losses[3]:.4f}"
                        f"  contractive: {losses[4]:.4f}"
                        f"  margin: {losses[5]:.4f}"
                        f"  β={beta:.3f}  γ={gamma:.3f}"
                    )
                    if validation_dataset is not None:
                        df_val = self._run_validation(
                            validation_dataset, validation_repetition_size
                        )
                        df_val = df_val.T.rename(index={0: step})
                        history_valid.append(df_val)
                        log += (
                            f"  val_valid: {df_val.loc[step, 'VALID_RATIO']:.3f}"
                            f"  val_sim: {df_val.loc[step, 'AVERAGE_SIMILARITY']:.3f}"
                        )
                    print(log)

                if step >= total_steps:
                    break
                step += 1

        df_history = pd.DataFrame(
            history,
            columns=[
                "LOSS_TOTAL",
                "LOSS_RECONSTRUCTION_LEAD",
                "LOSS_RECONSTRUCTION_OPT",
                "LOSS_RECONSTRUCTION_CONTROL",
                "LOSS_CONTRACTIVE",
                "LOSS_MARGIN",
                "BETA",
                "GAMMA",
            ],
        )
        df_history_valid = pd.concat(history_valid) if history_valid else pd.DataFrame()
        return df_history, df_history_valid

    def finetune(
        self,
        dataset: TripletSmilesDataset,
        reward_fn: RewardFunction,
        *,
        batch_size: int = 5,
        total_steps: int = 5000,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.995,
        buffer_size: int = 10,
        buffer_batch_size: int = 10,
        validation_dataset: Optional[ValidationSmilesDataset] = None,
        validation_repetition_size: int = 5,
        checkpoint_step: int = 500,
        checkpoint_path: Optional[str | Path] = None,
        display_step: int = 50,
        validation_step: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Policy-gradient RL fine-tuning with multi-objective reward.

        Reward = 0.4 × KRAS_affinity + 0.4 × BBBp + 0.2 × QED,
        given Tanimoto(source, generated) > 0.4.
        """
        use_cuda = torch.cuda.is_available()
        history: List[Tuple] = []
        history_valid: List[pd.DataFrame] = []
        replay_buffer = ReplayBuffer()
        start_time = time.time()

        for step in range(1, total_steps + 1):
            # ---- Fill replay buffer ----
            for batch in DataLoader(
                dataset, batch_size=1, shuffle=True,
                drop_last=False, pin_memory=use_cuda,
            ):
                enc_src = dataset.encode(batch["smiles_s"], int(batch["length_s"].max()))
                len_src = batch["length_s"]

                enc_src_rep = enc_src.repeat(buffer_batch_size, 1)
                len_src_rep = len_src.repeat(buffer_batch_size)

                enc_generated = self.generate(enc_src_rep, len_src_rep)
                smi_src = batch["smiles_s"][0]

                generated_smiles = [
                    dataset.vocab.SOS_CHAR + smi + dataset.vocab.EOS_CHAR
                    for smi in dataset.decode(enc_generated)
                ]
                for smi_tar in generated_smiles:
                    reward, sim, prop = reward_fn(smi_src[1:-1], smi_tar[1:-1])
                    if reward > 0:
                        replay_buffer.push(smi_src, smi_tar, reward, sim, prop)

                # Also replay the known Opt (supervised signal)
                smi_opt = batch["smiles_t"][0]
                reward, sim, prop = reward_fn(smi_src[1:-1], smi_opt[1:-1])
                replay_buffer.push(smi_src, smi_opt, reward, sim, prop)

                if len(replay_buffer) >= buffer_size:
                    break

            avg_reward, avg_sim, avg_prop = replay_buffer.statistics()

            # ---- Policy gradient update ----
            for batch in DataLoader(
                replay_buffer, batch_size=batch_size, shuffle=True,
                drop_last=False, pin_memory=False,
            ):
                enc_src = dataset.encode(batch["smiles_src"], int(batch["length_src"].max()))
                len_src = batch["length_src"]
                enc_tar = dataset.encode(batch["smiles_tar"], int(batch["length_tar"].max()))
                len_tar = batch["length_tar"]
                rewards = batch["reward"].to(self.device)

                rl_loss = self._policy_gradient_step(
                    enc_src, len_src, enc_tar, len_tar, rewards,
                    lr=learning_rate, discount_factor=discount_factor,
                )
                replay_buffer.commit_pops()
                break

            history.append((rl_loss, avg_reward, avg_sim, avg_prop))

            if checkpoint_path is not None and step % checkpoint_step == 0:
                self.save_weights(checkpoint_path)

            if step % display_step == 0:
                elapsed = (time.time() - start_time) / 60
                log = (
                    f"[{step:06d}/{total_steps:06d}]"
                    f"  loss: {rl_loss:.4f}"
                    f"  reward: {avg_reward:.4f}"
                    f"  similarity: {avg_sim:.4f}"
                    f"  property: {avg_prop:.4f}"
                )
                if validation_dataset is not None and step % validation_step == 0:
                    df_val_gen = self.transform_dataset(
                        validation_dataset, n_samples=validation_repetition_size
                    )
                    records = []
                    for smi_src, smi_tar in df_val_gen.values:
                        _, sim_v, prop_v = reward_fn(smi_src, smi_tar)
                        records.append((smi_src, smi_tar, sim_v, prop_v))
                    df_metrics = evaluate_metric_validation(
                        pd.DataFrame.from_records(records),
                        num_decode=validation_repetition_size,
                    )
                    df_metrics = df_metrics.T.rename(index={0: step})
                    history_valid.append(df_metrics)
                    log += (
                        f"  val_valid: {df_metrics.loc[step, 'VALID_RATIO']:.3f}"
                        f"  val_sim: {df_metrics.loc[step, 'AVERAGE_SIMILARITY']:.3f}"
                        f"  val_prop: {df_metrics.loc[step, 'AVERAGE_PROPERTY']:.3f}"
                    )
                log += f"  ({elapsed:.1f} min)"
                print(log)

        df_history = pd.DataFrame(
            history, columns=["LOSS", "REWARD", "SIMILARITY", "PROPERTY"]
        )
        df_history_valid = (
            pd.concat(history_valid) if history_valid else pd.DataFrame()
        )
        return df_history, df_history_valid

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform_dataset(
        self,
        dataset: ValidationSmilesDataset,
        n_samples: int = 20,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Generate *n_samples* molecules for every molecule in *dataset*.

        Returns
        -------
        pd.DataFrame with columns [source_smiles, generated_smiles].
        """
        use_cuda = torch.cuda.is_available()
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            drop_last=False, pin_memory=use_cuda,
        )
        if show_progress:
            loader = tqdm.tqdm(loader, total=len(dataset))

        records: List[Tuple[str, str]] = []
        for batch in loader:
            enc = dataset.encode(batch["smiles_s"], int(batch["length_s"].max()))
            length = batch["length_s"]
            for _ in range(n_samples):
                generated = self.generate(enc, length)
                smi = dataset.decode(generated)[0]
                records.append((batch["smiles_s"][0][1:-1], smi))

        return pd.DataFrame.from_records(records, columns=["source", "generated"])

    @torch.no_grad()
    def generate(self, smiles_encoded: Tensor, lengths: Tensor, max_len: int = 128) -> np.ndarray:
        """Generate SMILES from encoded source molecules.

        Parameters
        ----------
        smiles_encoded:
            Integer token tensor of shape (batch, seq).
        lengths:
            Sequence lengths of shape (batch,).
        max_len:
            Maximum generation length.

        Returns
        -------
        np.ndarray of shape (batch, max_len) with generated token indices.
        """
        self.eval()
        batch_size = smiles_encoded.size(0)
        mean, log_var = self.encoder(smiles_encoded, lengths)
        z = self.encoder.sample_latent(mean, log_var)

        all_generated = []
        for i in range(batch_size):
            z_i = z[i].unsqueeze(0)
            seq = self._decode_greedy_sample(z_i, max_len)
            all_generated.append(seq.cpu().numpy())

        return np.concatenate(all_generated, axis=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)

    def save_config(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"VOCAB_SIZE,{self.vocab_size}\n")
            f.write(f"HIDDEN_SIZE,{self.hidden_size}\n")
            f.write(f"LATENT_SIZE,{self.latent_size}\n")
            f.write(f"NUM_LAYERS,{self.num_layers}\n")
            f.write(f"DROPOUT,{self.dropout}\n")
            f.write(f"SOS_IDX,{self.sos_idx}\n")
            f.write(f"EOS_IDX,{self.eos_idx}\n")
            f.write(f"PAD_IDX,{self.pad_idx}\n")

    def _load_config(self, path: str | Path) -> None:
        params: Dict[str, str] = {}
        with open(path) as f:
            for line in f:
                k, v = line.strip().split(",", 1)
                params[k] = v
        self.vocab_size = int(params["VOCAB_SIZE"])
        self.hidden_size = int(params["HIDDEN_SIZE"])
        self.latent_size = int(params["LATENT_SIZE"])
        self.num_layers = int(params["NUM_LAYERS"])
        self.dropout = float(params["DROPOUT"])
        self.sos_idx = int(params["SOS_IDX"])
        self.eos_idx = int(params["EOS_IDX"])
        self.pad_idx = int(params["PAD_IDX"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pretrain_step(
        self,
        enc_lead: Tensor, len_lead: Tensor,
        enc_opt: Tensor, len_opt: Tensor,
        enc_con: Tensor, len_con: Tensor,
        lr: float, beta: float, gamma: float,
    ) -> Tuple[float, ...]:
        """Single contrastive pretraining step."""
        self.train()
        opt_enc = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        opt_dec = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        opt_enc.zero_grad()
        opt_dec.zero_grad()

        mean_l, logvar_l = self.encoder(enc_lead, len_lead)
        mean_o, logvar_o = self.encoder(enc_opt, len_opt)
        mean_c, logvar_c = self.encoder(enc_con, len_con)

        z_l = self.encoder.sample_latent(mean_l, logvar_l)
        z_o = self.encoder.sample_latent(mean_o, logvar_o)
        z_c = self.encoder.sample_latent(mean_c, logvar_c)

        logits_l = self.decoder(enc_lead, z_l)
        logits_o = self.decoder(enc_opt, z_o)
        logits_c = self.decoder(enc_con, z_c)

        loss_recon_l = self._reconstruction_loss(logits_l, enc_lead)
        loss_recon_o = self._reconstruction_loss(logits_o, enc_opt)
        loss_recon_c = self._reconstruction_loss(logits_c, enc_con)

        loss_contractive = self._frechet_distance(mean_l, logvar_l, mean_o, logvar_o)
        loss_margin = (
            self._margin_loss(mean_l, mean_c)
            + self._margin_loss(mean_o, mean_c)
        )

        loss = (
            0.3 * loss_recon_l
            + 0.3 * loss_recon_o
            + 0.4 * loss_recon_c
            + beta * loss_contractive
            + gamma * loss_margin
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        opt_enc.step()
        opt_dec.step()

        return (
            loss.item(),
            loss_recon_l.item(),
            loss_recon_o.item(),
            loss_recon_c.item(),
            loss_contractive.item(),
            loss_margin.item(),
        )

    def _policy_gradient_step(
        self,
        enc_src: Tensor, len_src: Tensor,
        enc_tar: Tensor, len_tar: Tensor,
        rewards: Tensor,
        lr: float,
        discount_factor: float,
    ) -> float:
        """Single policy-gradient update step."""
        batch_size = enc_tar.size(0)
        seq_len = enc_tar.size(1)

        self.encoder.eval()
        self.decoder.train()

        opt_dec = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        opt_dec.zero_grad()

        with torch.no_grad():
            mean_src, logvar_src = self.encoder(enc_src, len_src)
        z_src = self.encoder.sample_latent(mean_src, logvar_src)

        logits_tar = self.decoder(enc_tar, z_src)
        log_probs = torch.nn.functional.log_softmax(logits_tar, dim=-1)

        # Discounted cumulative returns G_t
        returns = torch.zeros(batch_size, seq_len, device=self.device)
        returns[torch.arange(batch_size), len_tar - 1] = rewards
        for t in range(1, seq_len):
            returns[:, -t - 1] = returns[:, -t - 1] + returns[:, -t] * discount_factor

        weighted_log_probs = returns.unsqueeze(-1) * log_probs
        target_flat = enc_tar[:, 1:].contiguous().view(-1)
        wlp_flat = weighted_log_probs[:, :-1, :].contiguous().view(-1, weighted_log_probs.size(-1))

        rl_loss = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(wlp_flat, target_flat)
        rl_loss.backward()
        nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        opt_dec.step()

        return rl_loss.item()

    def _decode_greedy_sample(self, z: Tensor, max_len: int) -> Tensor:
        """Autoregressive decode using multinomial sampling from softmax."""
        batch_size = z.size(0)
        outs = torch.full(
            (batch_size, max_len), self.pad_idx, dtype=torch.long, device=self.device
        )
        hidden = self.decoder.init_hidden(batch_size)
        token = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device
        )

        for t in range(max_len):
            if token[0, 0].item() in (self.eos_idx, self.pad_idx):
                outs[:, t] = self.eos_idx
                break
            outs[:, t] = token.squeeze(1)
            logits, hidden = self.decoder(token, z, hidden)
            probs = torch.softmax(logits.squeeze(1), dim=-1)
            token = torch.multinomial(probs, num_samples=1)

        return outs

    def _run_validation(
        self,
        dataset: ValidationSmilesDataset,
        n_samples: int,
    ) -> pd.DataFrame:
        from galaxyair.utils.molecular import compute_tanimoto_similarity
        df_gen = self.transform_dataset(dataset, n_samples=n_samples, show_progress=False)
        records = [
            (src, tar, compute_tanimoto_similarity(src, tar), 0.999)
            for src, tar in df_gen.values
        ]
        return evaluate_metric_validation(
            pd.DataFrame.from_records(records), num_decode=n_samples
        )

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _reconstruction_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_flat = target[:, 1:].contiguous().view(-1)
        lp_flat = log_probs[:, :-1, :].contiguous().view(-1, log_probs.size(2))
        return nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(lp_flat, target_flat)

    def _margin_loss(self, mean_a: Tensor, mean_b: Tensor) -> Tensor:
        """Push mean_b away from mean_a in latent space."""
        dist_sq = (mean_a - mean_b).pow(2).sum(1)
        return torch.nn.functional.softplus(1.0 - dist_sq).mean()

    def _frechet_distance(
        self,
        mean_a: Tensor, logvar_a: Tensor,
        mean_b: Tensor, logvar_b: Tensor,
    ) -> Tensor:
        """Fréchet / Wasserstein-2 distance between two Gaussians."""
        loss = (mean_a - mean_b).pow(2)
        loss = loss + torch.exp(logvar_a) + torch.exp(logvar_b)
        loss = loss - 2.0 * torch.exp(0.5 * (logvar_a + logvar_b))
        return loss.sum(1).mean()
