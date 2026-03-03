import os
import sys
import tensorflow as tf
from keras import backend as K

sys.path.append('/home/data-house-01/guiyike/BBBmodel')

from ShengXiaBBB1 import *

predictor = Predictor(config, token_table, model_type, descriptor)


def get_bbb_score(smile):
    prediction0 = predictor.predict([smile], data_utils)
    #print(smile)
    #print(prediction0[0])
    return float(prediction0[0])

import pandas as pd

# Read the CSV file
df = pd.read_csv('yike.csv')

# Create an empty list to store the BBB scores
scores = []

# Iterate over the 'smiles' column
for smile in df['SMILES']:
    # Call the get_bbb_score function and append the score to the list
    score = get_bbb_score(smile)
    scores.append(score)

# Add a new column 'score' to the DataFrame
df['score'] = scores

# Save the modified DataFrame to a new CSV file
df.to_csv('yike_test.csv', index=False)