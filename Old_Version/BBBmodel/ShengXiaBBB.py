from utils import *
from tokens import tokens_table
from tensorflow.keras.models import model_from_json
import numpy as np
from utils import *

config_file = '/home/data-house-01/zhangxiang/BBBmodel/configPredictor.json'  # Name of the configuration file
property_identifier = 'bbb'
model_type = 'dnn'  # 'dnn', 'SVR', 'RF', or 'KNN'
descriptor = 'SMILES'  # The type of model's descriptor can be 'SMILES' or 'ECFP'. If we want to use
# rnn architecture we use SMILES. Conversely, if we want to use a fully connected architecture, we use ECFP descriptors.
searchParameters = False  # True (gridSearch) or False (train with the optimal parameters)
# load model configurations

# load model configurations
config = load_config(config_file, property_identifier)
directories([config.checkpoint_dir])

# Load the table of possible tokens
token_table = tokens_table().table

# Read and extract smiles and labels from the csv file
smiles_raw, labels_raw = reading_csv(config, property_identifier)

# Padd each SMILES string with spaces until reaching the size of the largest molecule
smiles_padded, padd = pad_seq(smiles_raw, token_table, 0)
config.paddSize = padd

# Compute the dictionary that makes the correspondence between each token and unique integers
tokenDict = smilesDict(token_table)

# Tokenize - transform the SMILES strings into lists of tokens
[tokens, problem_idx] = tokenize(smiles_padded, token_table)
labels_raw = np.delete(labels_raw, problem_idx).tolist()

# Transforms each token to the respective integer, according to the previously computed dictionary
smiles_int = smiles2idx(tokens, tokenDict)

data_rnn_smiles = data_division(config, smiles_int, labels_raw, True, model_type, descriptor)
x_test = data_rnn_smiles[2]
y_test = data_rnn_smiles[3]
data_cv = cv_split(data_rnn_smiles, config)

data_utils = []
metrics = []
for split in data_cv:
    data_i = []
    train, val = split

    X_train = data_rnn_smiles[0][train]
    y_train = np.array(data_rnn_smiles[1])[train]
    X_val = data_rnn_smiles[0][val]
    y_val = np.array(data_rnn_smiles[1])[val]
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    data_i.append(X_train)
    data_i.append(y_train)
    data_i.append(x_test)
    data_i.append(y_test)
    data_i.append(X_val)
    data_i.append(y_val)

    data_i, data_aux = normalize(data_i)

    data_utils.append(data_aux)




class Predictor(object):
    def __init__(self, config, tokens, model_type, descriptor_type):
        super(Predictor, self).__init__()
        self.tokens = tokens
        self.config = config
        self.model_type = model_type
        self.descriptor_type = descriptor_type
        loaded_models = []
        for i in range(5):
            json_file = open(
                "/home/data-house-01/zhangxiang/BBBmodel/experiments/bbb-smiles/Model/" + "model" + str(i) + ".json",
                'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(
                "/home/data-house-01/zhangxiang/BBBmodel/experiments/bbb-smiles/Model/" + "model" + str(i) + ".h5")
            print("Model " + str(i) + " loaded from disk!")
            loaded_models.append(loaded_model)

        self.loaded_models = loaded_models

    def predict(self, smiles, data):
        smiles_padded, kl = pad_seq(smiles, self.tokens, self.config.paddSize)
        d = smilesDict(self.tokens)
        tokenized_smiles = tokenize(smiles_padded, self.tokens)
        tokenized_smiles_new = [tokenized_smiles[0][0], []]
        data_2_predict = smiles2idx(tokenized_smiles_new, d)

        prediction = []
        for m in range(len(self.loaded_models)):
            prediction.append(self.loaded_models[m].predict(data_2_predict))
        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        prediction = denormalization(prediction, data)
        prediction = np.mean(prediction, axis=0)

        return prediction


list_ss = [["NC(N)=NC(=O)c1nc(Cl)c(N)nc1N"]]
predictor = Predictor(config, token_table, model_type, descriptor)
prediction0 = predictor.predict(list_ss[0], data_utils)
print(prediction0[0])

# prediction0 = predictor.predict(list_ss[1], data_utils)
# print(prediction0[1])
#
# prediction0 = predictor.predict(list_ss[2], data_utils)
# print(prediction0[2])
