import os
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
import random


class data_processing(object):
    def __init__(
            self,
            cell_id_list,
            aux_targets,
            aux_tf_dict,
            output_size=[1],
            im_size=None):
        """Allen cell tfrecord global variable init."""
        self.name = 'allen_cell_%s' % cell_id_list
        self.config = Config()
        self.folds = {
            'train': 'training',
            'test': 'testing'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'f': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'f': tf_fun.fixed_len_feature(dtype='float')
        }
        # Add aux data
        for ((tk, tv), (dk, dv)) in zip(
                aux_targets.iteritems(),
                aux_tf_dict.iteritems()):
            self.targets[tk] = tv
            self.tf_dict[dk] = dv

        self.output_size = output_size
        self.im_size = im_size
        self.preprocess = [None]
        self.shuffle = False  # Preshuffle data?

    def get_data(self):
        """Main method for packaging data."""
        files = self.get_files()
        labels = self.get_labels(files)
        return files, labels

    def get_files(self):
        files = {}
        for k, fold in self.folds.iteritems():
            it_files = []
            dirs = glob(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    fold,
                    '*'))
            for d in dirs:
                it_files += [glob(
                    os.path.join(
                        d,
                        '*%s' % self.extension))]
            it_files = py_utils.flatten_list(it_files)
            if self.shuffle:
                random.shuffle(it_files)
            files[k] = it_files
        return files

    def get_labels(self, files):
        labels = {}
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [int(f.split('/')[-2])]
            labels[k] = it_labels
        return labels
