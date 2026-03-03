"""
SMILES dataset classes for the VAE generator.

Data format
-----------
Triplet file (train):  space-separated, no header
    <src_smiles> <tar_smiles> <neg_smiles>

Validation file:  one SMILES per line, no header.

Special tokens
--------------
SOS : '<'    (start-of-sequence)
EOS : '>'    (end-of-sequence)
PAD : '_'    (padding)
UNK : '?'    (unknown character)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class SmilesVocabulary:
    """Character-level vocabulary for SMILES strings.

    Parameters
    ----------
    chars:
        Iterable of characters to include.  The four special tokens are
        always added automatically.
    """

    SOS_CHAR = "<"
    EOS_CHAR = ">"
    PAD_CHAR = "_"
    UNK_CHAR = "?"

    def __init__(self, chars: Optional[List[str]] = None) -> None:
        special = [self.PAD_CHAR, self.SOS_CHAR, self.EOS_CHAR, self.UNK_CHAR]
        all_chars = special + sorted(set(chars or []) - set(special))
        self._char_to_idx: Dict[str, int] = {c: i for i, c in enumerate(all_chars)}
        self._idx_to_char: Dict[int, str] = {i: c for c, i in self._char_to_idx.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._char_to_idx)

    @property
    def pad_idx(self) -> int:
        return self._char_to_idx[self.PAD_CHAR]

    @property
    def sos_idx(self) -> int:
        return self._char_to_idx[self.SOS_CHAR]

    @property
    def eos_idx(self) -> int:
        return self._char_to_idx[self.EOS_CHAR]

    @property
    def unk_idx(self) -> int:
        return self._char_to_idx[self.UNK_CHAR]

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode_char(self, char: str) -> int:
        return self._char_to_idx.get(char, self.unk_idx)

    def decode_idx(self, idx: int) -> str:
        return self._idx_to_char.get(idx, self.UNK_CHAR)

    def encode_smiles(self, smiles: str) -> List[int]:
        return [self.encode_char(c) for c in smiles]

    def decode_indices(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            char = self.decode_idx(idx)
            if char in (self.EOS_CHAR, self.PAD_CHAR):
                break
            if char == self.SOS_CHAR:
                continue
            chars.append(char)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for char, idx in sorted(self._char_to_idx.items(), key=lambda x: x[1]):
                writer.writerow([char, idx])

    @classmethod
    def load(cls, path: str | Path) -> "SmilesVocabulary":
        char_to_idx: Dict[str, int] = {}
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                char_to_idx[row[0]] = int(row[1])
        vocab = cls.__new__(cls)
        vocab._char_to_idx = char_to_idx
        vocab._idx_to_char = {i: c for c, i in char_to_idx.items()}
        return vocab


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _wrap_with_tokens(smiles: str, sos: str, eos: str) -> str:
    return sos + smiles + eos


def _encode_batch(
    smiles_list: List[str],
    vocab: SmilesVocabulary,
    max_len: int,
    device: torch.device,
) -> Tensor:
    """Encode a list of SMILES strings into a zero-padded integer tensor.

    Parameters
    ----------
    smiles_list:
        List of raw (already token-wrapped) SMILES strings.
    vocab:
        Vocabulary used for encoding.
    max_len:
        Length to pad/truncate to.
    device:
        Target device for the returned tensor.

    Returns
    -------
    Tensor of shape (batch, max_len) on *device*.
    """
    batch_size = len(smiles_list)
    out = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long, device=device)
    for i, smiles in enumerate(smiles_list):
        indices = vocab.encode_smiles(smiles)[:max_len]
        out[i, : len(indices)] = torch.tensor(indices, dtype=torch.long, device=device)
    return out


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class TripletSmilesDataset(Dataset):
    """Dataset of (source, target, negative) SMILES triplets for pretraining.

    The file must be space-separated with no header; each row contains three
    SMILES strings representing a matched pair and a negative example.

    Parameters
    ----------
    filepath:
        Path to the triplet text file.
    device:
        PyTorch device for encoded tensors.
    vocab_path:
        Optional path to a pre-built vocabulary CSV.  When None a new
        vocabulary is built from the data.
    """

    def __init__(
        self,
        filepath: str | Path,
        device: torch.device,
        vocab_path: Optional[str | Path] = None,
    ) -> None:
        self.device = device
        self._records: List[tuple[str, str, str]] = []

        all_chars: List[str] = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                src, tar, neg = parts
                self._records.append((src, tar, neg))
                all_chars.extend(list(src + tar + neg))

        if vocab_path is not None:
            self.vocab = SmilesVocabulary.load(vocab_path)
        else:
            self.vocab = SmilesVocabulary(chars=all_chars)

    def save_vocab(self, path: str | Path) -> None:
        self.vocab.save(path)

    def encode(self, smiles_list: List[str], max_len: int) -> Tensor:
        return _encode_batch(smiles_list, self.vocab, max_len, self.device)

    def decode(self, encoded: "np.ndarray") -> List[str]:
        import numpy as np
        result = []
        for row in encoded:
            result.append(self.vocab.decode_indices(row.tolist()))
        return result

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        src, tar, neg = self._records[idx]
        sos, eos = self.vocab.SOS_CHAR, self.vocab.EOS_CHAR
        src_tok = _wrap_with_tokens(src, sos, eos)
        tar_tok = _wrap_with_tokens(tar, sos, eos)
        neg_tok = _wrap_with_tokens(neg, sos, eos)
        return {
            "smiles_s": src_tok,
            "length_s": torch.tensor(len(src_tok), dtype=torch.long),
            "smiles_t": tar_tok,
            "length_t": torch.tensor(len(tar_tok), dtype=torch.long),
            "smiles_n": neg_tok,
            "length_n": torch.tensor(len(neg_tok), dtype=torch.long),
        }


class ValidationSmilesDataset(Dataset):
    """Single-SMILES dataset for validation and generation.

    Parameters
    ----------
    filepath:
        Path to a plain text file with one SMILES per line.
    vocab_path:
        Path to an existing vocabulary CSV (created during training).
    device:
        PyTorch device for encoded tensors.
    """

    def __init__(
        self,
        filepath: str | Path,
        vocab_path: str | Path,
        device: torch.device,
    ) -> None:
        self.device = device
        self.vocab = SmilesVocabulary.load(vocab_path)
        self._smiles: List[str] = []
        with open(filepath, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    self._smiles.append(s)

    def encode(self, smiles_list: List[str], max_len: int) -> Tensor:
        return _encode_batch(smiles_list, self.vocab, max_len, self.device)

    def decode(self, encoded: "np.ndarray") -> List[str]:
        result = []
        for row in encoded:
            result.append(self.vocab.decode_indices(row.tolist()))
        return result

    def __len__(self) -> int:
        return len(self._smiles)

    def __getitem__(self, idx: int) -> Dict:
        sos, eos = self.vocab.SOS_CHAR, self.vocab.EOS_CHAR
        smiles = _wrap_with_tokens(self._smiles[idx], sos, eos)
        return {
            "smiles_s": smiles,
            "length_s": torch.tensor(len(smiles), dtype=torch.long),
        }
