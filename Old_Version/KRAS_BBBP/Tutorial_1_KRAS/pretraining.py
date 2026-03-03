import os
import sys
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt

sys.path = [os.path.abspath(os.path.join(os.getcwd(), os.pardir))] + sys.path
from COMA.dataset import TrainingSmilesDataset, ValidationSmilesDataset
from COMA.vae import SmilesAutoencoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print(device)
PROPERTY_NAME = "drd2"
input_dir = os.path.abspath(os.path.join(os.pardir, "DATA", PROPERTY_NAME))
output_dir = f"outputs_1-1_{PROPERTY_NAME.upper()}_pretraining_1024"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filepath_train = os.path.join(input_dir, "rdkit_train_triplet_1024.txt")
filepath_valid = os.path.join(input_dir, "rdkit_valid_1024.txt")
filepath_char2idx = os.path.join(output_dir, "char2idx.csv")
filepath_configs = os.path.join(output_dir, "configs.csv")
filepath_checkpoint = os.path.join(output_dir, "checkpoints.pt")
filepath_history = os.path.join(output_dir, "history.csv")
filepath_history_valid = os.path.join(output_dir, "history_valid.csv")

dataset = TrainingSmilesDataset(filepath_train, device=device)
dataset.save_char2idx(filepath_char2idx)
dataset_valid = ValidationSmilesDataset(filepath_valid, filepath_char2idx, device=device)

## Model configuration
model_configs = {"hidden_size": 128,
                 "latent_size": 128,
                 "num_layers": 2,
                 "vocab_size": dataset.vocab_size,
                 "sos_idx": dataset.sos_idx,
                 "eos_idx": dataset.eos_idx,
                 "pad_idx": dataset.pad_idx,
                 "device": device
                 }

## Model initialization
generator = SmilesAutoencoder(**model_configs)

## Configuration save
generator.save_config(filepath_configs)

df_history, df_history_valid = generator.fit(dataset,
                                             validation_dataset=dataset_valid,
                                             checkpoint_filepath=filepath_checkpoint)

df_history.to_csv(filepath_history, index=False)
df_history_valid.to_csv(filepath_history_valid, index=False)

fig, axes = plt.subplots(4, 1, figsize=(8, 8))

axes[0].plot(df_history.loc[:, "LOSS_TOTAL"], label="Total loss")
axes[1].plot(df_history.loc[:, "LOSS_RECONSTRUCTION_SOURCE"], label="Recon. loss (src)")
axes[1].plot(df_history.loc[:, "LOSS_RECONSTRUCTION_TARGET"], label="Recon. loss (tar)")
axes[1].plot(df_history.loc[:, "LOSS_RECONSTRUCTION_NEGATIVE"], label="Recon. loss (neg)")
axes[2].plot(df_history.loc[:, "LOSS_CONTRACTIVE"], label="Contractive loss (src,tar)")
axes[3].plot(df_history.loc[:, "LOSS_MARGIN"], label="Margin loss (src,tar,neg)")

axes[3].set_xlabel("Iteration")
for ax in axes:
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig('/home/house5/wangyuqing/COMA/Tutorial_1_DRD2/loss_1024.png')
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(8, 4))

axes[0].plot(df_history_valid.loc[:, "VALID_RATIO"], label="Validity")
axes[1].plot(df_history_valid.loc[:, "AVERAGE_SIMILARITY"], label="Tanimoto coeff.")

axes[1].set_xlabel("Iteration")
for ax in axes:
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig('/home/house5/wangyuqing/COMA/Tutorial_1_DRD2/Validity_1024.png')
plt.close()
