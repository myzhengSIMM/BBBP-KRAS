import sys

sys.path.append('/home/house/wangyuqing/CCR5model')

from ShengXiaAffinity import *

predictor = Predictor(config, token_table, model_type, descriptor)


def map_to_zero_one_interval(x, interval):
    a, b = interval[0], interval[1]
    return (x - a) / (b - a)


interval = [4.75, 9.1]
data_utils = [[9.077, 4.988], [9.096, 4.844], [9.08, 4.8455], [9.1, 4.7885], [9.1, 4.89]]


def get_score(smile):
    prediction0 = predictor.predict([smile], data_utils)
    mapped_value = map_to_zero_one_interval(prediction0[0], interval)
    return float(mapped_value)
    # return float(prediction0[0])


def get_ccr5_score(smile):
    prediction0 = predictor.predict([smile], data_utils)
    return float(prediction0[0])
