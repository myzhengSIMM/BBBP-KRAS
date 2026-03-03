import os
import sys
import tensorflow as tf
from keras import backend as K

sys.path.append('/home/house/wangyuqing/KORmodel')

from ShengXiaLogD import *

predictor = Predictor(config, token_table, model_type, descriptor)


def map_to_zero_one_interval(x, interval):
    a, b = interval[0], interval[1]
    return (x - a) / (b - a)


interval = [-0.15, 3.6]
data_utils = [[3.6, -0.1105], [3.6, -0.12], [3.6, -0.121], [3.6, -0.12], [3.6, -0.15]]


def get_logd_score(smile):
    prediction0 = predictor.predict([smile], data_utils)
    #mapped_value = map_to_zero_one_interval(prediction0[0], interval)
    return float(prediction0[0])

