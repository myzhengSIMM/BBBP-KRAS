import os
import sys
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt

sys.path = [os.path.abspath(os.path.join(os.getcwd(), os.pardir))] + sys.path
from COMA.dataset import TrainingSmilesDataset, ValidationSmilesDataset
from COMA.vaenew import SmilesAutoencoder, RewardFunction
from COMA.properties import drd2, similarity

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
PROPERTY_NAME = "drd2"

SCORING_PROPERTY_FT = drd2
SCORING_TANIMOTO_FT = similarity

## Configure the parameters of a reward function for the target property
threshold_property = 0.
threshold_similarity = 0.4

input_data_dir = os.path.abspath(os.path.join(os.pardir, "DATA", PROPERTY_NAME))
input_ckpt_dir = f"outputs_1-1_{PROPERTY_NAME.upper()}_pretraining_1024"
output_dir = f"outputs_1-2_{PROPERTY_NAME.upper()}_finetuning_1029"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filepath_train = os.path.join(input_data_dir, "rdkit_train_triplet_1024.txt")
filepath_valid = os.path.join(input_data_dir, "rdkit_valid_1024.txt")
filepath_pretrain_ckpt = os.path.join(input_ckpt_dir, "checkpoints.pt")
filepath_pretrain_configs = os.path.join(input_ckpt_dir, "configs.csv")
filepath_pretrain_char2idx = os.path.join(input_ckpt_dir, "char2idx.csv")
filepath_char2idx = os.path.join(output_dir, "char2idx.csv")
filepath_configs = os.path.join(output_dir, "configs.csv")
filepath_checkpoint = os.path.join(output_dir, "checkpoints.pt")
filepath_history = os.path.join(output_dir, "history.csv")
filepath_history_valid = os.path.join(output_dir, "history_valid.csv")

dataset = TrainingSmilesDataset(filepath_train, filepath_char2idx=filepath_pretrain_char2idx, device=device)
dataset.save_char2idx(filepath_char2idx)
dataset_valid = ValidationSmilesDataset(filepath_valid, filepath_char2idx, device=device)

## Model configuration
model_configs = {"hidden_size": None,
                 "latent_size": None,
                 "num_layers": None,
                 "vocab_size": None,
                 "sos_idx": None,
                 "eos_idx": None,
                 "pad_idx": None,
                 "device": device,
                 "filepath_config": filepath_pretrain_configs}

## Model initialization
generator = SmilesAutoencoder(**model_configs)

## Load pretrained model
generator.load_model(filepath_pretrain_ckpt)

## Configuration save
generator.save_config(filepath_configs)

reward_ft = RewardFunction(similarity_ft=SCORING_TANIMOTO_FT,
                           scoring_ft=SCORING_PROPERTY_FT,
                           threshold_property=threshold_property,
                           threshold_similarity=threshold_similarity)

df_history, df_history_valid = generator.policy_gradient(dataset, reward_ft,
                                                         validation_dataset=dataset_valid,
                                                         checkpoint_filepath=filepath_checkpoint)

df_history.to_csv(filepath_history, index=False)
df_history_valid.to_csv(filepath_history_valid, index=False)