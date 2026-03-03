import os
import tensorflow as tf
from keras import backend as K
from utils import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from imblearn.over_sampling import SMOTE
from tokens import tokens_table
from tensorflow.keras.models import model_from_json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
session = tf.compat.v1.Session()
K.set_session(session)

config_file = '/home/data-house-01/guiyike/BBBmodel/configPredictor.json'  # Name of the configuration file
property_identifier = 'bbb'
model_type = 'dnn'  # 'dnn', 'SVR', 'RF', or 'KNN'
descriptor = 'ECFP'  # The type of model's descriptor can be 'SMILES' or 'ECFP'. If we want to use
# rnn architecture we use SMILES. Conversely, if we want to use a fully connected architecture, we use ECFP descriptors.
searchParameters = False  # True (gridSearch) or False (train with the optimal parameters)

config = load_config(config_file, property_identifier)
directories([config.checkpoint_dir])

# Load the table of possible tokens
token_table = tokens_table().table

# Read and extract smiles and labels from the csv file
smiles_raw, labels_raw = reading_csv(config, property_identifier)
# print(len(smiles_raw))
mols = [Chem.MolFromSmiles(x) for x in smiles_raw]

morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits = 2048) for x in mols]


# convert the RDKit explicit vectors into numpy arrays
morg_fp_np = []
for fp in morgan_fp:
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    morg_fp_np.append(arr)


x_morg = morg_fp_np

x_morg_rsmp, y_morg_rsmp = SMOTE().fit_resample(x_morg, labels_raw)

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

data_utils = [[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]]


def SMILES2ECFP(smiles, radius=3, bit_len=2048, index=None):
    """
    This function transforms a list of SMILES strings into a list of ECFP with
    radius 3.
    ----------
    smiles: List of SMILES strings to transform
    Returns
    -------
    This function return the SMILES strings transformed into a vector of 4096 elements
    """
    fps = np.zeros((len(smiles), bit_len))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        try:

            mol = MurckoScaffold.GetScaffoldForMol(mol)

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:
            print(smile)
            fps[i, :] = [0] * bit_len
    return pd.DataFrame(fps, index=(smiles if index is None else index))


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
                "/home/data-house-01/guiyike/BBBmodel/experiments/bbb-kras/Model/" + "model" + str(i) + ".json",
                'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(
                "/home/data-house-01/guiyike/BBBmodel/experiments/bbb-kras/Model/" + "model" + str(i) + ".h5")
            print("Model " + str(i) + " loaded from disk!")
            loaded_models.append(loaded_model)

        self.loaded_models = loaded_models

    def predict(self, smiles, data):
        data_2_predict = SMILES2ECFP(smiles)
        prediction = []
        for m in range(len(self.loaded_models)):
            prediction.append(self.loaded_models[m].predict(data_2_predict,verbose=0))
        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        prediction = denormalization(prediction, data)
        prediction = np.mean(prediction, axis=0)
        return prediction
