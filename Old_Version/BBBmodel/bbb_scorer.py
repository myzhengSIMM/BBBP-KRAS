import os
import sys
import tensorflow as tf
from keras import backend as K
from ShengXiaBBB1 import *

predictor = Predictor(config, token_table, model_type, descriptor)


def get_bbb_score(smile):
    prediction0 = predictor.predict([smile], data_utils)
    return float(prediction0[0])


# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('/mnt/sdb/ykgui/BBBmodel/data/Book1.csv')
#
# # Create an empty list to store the BBB scores
# scores = []
#
# # Iterate over the 'smiles' column
# for smile in df['smiles']:
#     # Call the get_bbb_score function and append the score to the list
#     score = get_bbb_score(smile)
#     scores.append(score)
#
# # Add a new column 'score' to the DataFrame
# df['yike'] = scores
#
# # Save the modified DataFrame to a new CSV file
# df.to_csv('/mnt/sdb/ykgui/BBBmodel/data/Book1.csv', index=False)

list_ss_1 = get_bbb_score("ClC1=[C@]([C@]2=C(C#CC)C=C(CN3CCN(C(C=C)=O)C[C@@H]3CO4)C4=C2F)C(O)=CC=C1")
print(list_ss_1)
list_ss_2 = get_bbb_score("CC1=[C@]([C@]2=CC=C(C(N3CCN(C(C=C)=O)C[C@H]3CO4)=O)C4=C2Cl)C5=C(C=C1)NN=C5")
print(list_ss_2)
list_ss_3 = get_bbb_score("ClC1=[C@]([C@]2=C(C#C)C=C(CN3CCN(C(C=C)=O)C[C@@H]3CO4)C4=C2F)C(O)=CC=C1")
print(list_ss_3)
list_ss_4 = get_bbb_score("CC1=[C@]([C@]2=C(F)C=C(CN3CCN(C(C=C)=O)C[C@@H]3CO4)C4=C2F)C(O)=CC=C1")
print(list_ss_4)
# list_ss_5 = get_bbb_score("ClC1=[C@]([C@]2=C(OCCN(C)C)C=C(CN3CCN(C(C=C)=O)C[C@@H]3CO4)C4=C2F)C(O)=CC=C1")
# print(list_ss_5)
# list_ss_6 = get_bbb_score("ClC1=[C@]([C@]2=C(C#N)C=C(CN3CCN(C(C=C)=O)C[C@@H]3CO4)C4=C2F)C(O)=CC=C1")
# print(list_ss_6)
# list_ss_7 = get_bbb_score("FC1=[C@]([C@]2=C(F)C3=C(C(N4C[C@@H](C)N(C(C=C)=O)C[C@@H]4CCO5)=NC=N3)C5=C2Cl)C(O)=CC=C1")
# print(list_ss_7)
# list_ss_8 = get_bbb_score("ClC1=CC=CC2=C1C(N3CCC(C(N4C[C@H](CC#N)N(C(C(F)=C)=O)CC4)=NC(OC[C@@H]5CCCN5C)=N6)=C6C3)=CC=C2")
# print(list_ss_8)
# list_ss_9 = get_bbb_score("OC1=C(C2=NC(N(C3=C(C(C)C)N=CC=C3C)C(N=C4N5[C@@H](C)CN(C(C=C)=O)CC5)=O)=C4C=C2F)C(F)=CC=C1")
# print(list_ss_9)
