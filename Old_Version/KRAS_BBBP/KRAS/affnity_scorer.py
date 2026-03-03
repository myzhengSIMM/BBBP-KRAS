import sys

sys.path.append('/home/data-house-01/guiyike/dgl-lifesci-master/examples/property_prediction/csv_data_configuration')
from regression_inference_yike import *


def map_to_zero_one_interval(x, interval):
    a, b = interval[0], interval[1]
    return (x - a) / (b - a)


interval = [3.28, 9.54]


def get_affnity_score(smile):
    output_yike = get_output_yike([smile])
    return float(output_yike)


def get_affnity_score_mapped(smile):
    output_yike = get_output_yike([smile])
    mapped_value = map_to_zero_one_interval(output_yike, interval)
    return float(mapped_value)
