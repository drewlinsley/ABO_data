import os
import re
import time
from datetime import datetime
import numpy as np


def get_dt_stamp():
    return re.split(
        '\.', str(datetime.now()))[0].replace(
        ' ',
        '_').replace(
        ':',
        '_').replace(
        '-',
        '_')


def flatten_list(l):
    """Flatten a list of lists."""
    return [val for sublist in l for val in sublist]


def import_module(dataset, model_dir='dataset_processing'):
    """Dynamically import a module."""
    return getattr(
        __import__(model_dir, fromlist=[dataset]), dataset)


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def save_npys(data, model_name, output_string):
    """Save key/values in data as numpys."""
    for k, v in data.iteritems():
        output = os.path.join(
            output_string,
            '%s_%s' % (model_name, k)
            )
        np.save(output, v)


def check_path(data_pointer, log, msg):
    if not os.path.exists(data_pointer):
        log.error(msg)


def ifloor(x):
    return np.floor(x).astype(np.int)


def iceil(x):
    return np.ceil(x).astype(np.int)


def timestamp():
    tmptime = time.strftime('%D_%H_%M_%S')
    return tmptime.replace('/', '_')
