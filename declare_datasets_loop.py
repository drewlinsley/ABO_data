"""Declare new allen datasets to be encoded as TFrecords."""
import re
import os
import sys
import math
import shutil
import encode_datasets
import json
import string
import sshtunnel
import random
import threading
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import numpy as np
import itertools as it
import pandas as pd
from allen_config import Allen_Brain_Observatory_Config
from data_db import data_db
from data_db import credentials
from glob import glob
from datetime import datetime
from argparse import ArgumentParser
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread


def query_neurons_rfs(queries, filter_by_stim, sessions):
    """Wrapper for querying neurons from DB.
    Parameters
    ----------
    queries : list of list of querying parameters
    filter_by_stim: list of strings
    sessions: list strings

    Returns
    -------
    all_data_dicts : list of dictionaries in lists
    """
    all_data_dicts = []
    for q in queries:
        it_query = data_db.get_cells_all_data_by_rf_and_stimuli(
            rfs=q,
            stimuli=filter_by_stim,
            sessions=sessions)
        all_data_dicts += it_query
    return all_data_dicts


def create_grid_queries(all_data_dicts, smod=2):
    """
    Derives coordinates for x_width by y_width grids of the receptive field.
    Written by MPC modified by Michele.

    Parameters
    ----------
    all_data_dicts : list of list of dictionaries of Allen neurons
    smod: int for stride

    Returns
    -------
    queries : list of dictionaries in lists
    """
    rf_width_y, rf_width_x = [], []
    rf_y, rf_x = [], []
    for dat in all_data_dicts:
        rf_width_y += [dat['on_width_y']]
        rf_width_x += [dat['on_width_x']]
        cre_line = dat['cre_line']
        structure = dat['structure']
        rf_y += [dat['on_center_y']]
        rf_x += [dat['on_center_x']]

    # Get 95th percentile x and y width
    y_width = int(np.ceil(np.percentile(rf_width_y, 95)))
    x_width = int(np.ceil(np.percentile(rf_width_x, 95)))

    # Stride of the neuron bins
    y_stride = int(y_width/smod)
    x_stride = int(x_width/smod)

    # Create queries that have width and height by the receptive field size
    y_limit = int(np.floor(np.max(rf_y)))
    x_limit = int(np.floor(np.max(rf_x)))
    queries = []
    for x1 in (range(0, x_limit, x_stride)):
        for y1 in range(0, y_limit, y_stride):

            # Add width to each x1 as long as it's not more than x_max
            x2 = x1 + x_width
            # Add width to each y1 as long as it's not more than y_max
            y2 = y1 + y_width

            # add new coordinate range to queries
            queries += [[{
                    'rf_coordinate_range': {  # Get all cells
                        'x_min': x1,
                        'x_max': x2,
                        'y_min': y1,
                        'y_max': y2,
                    },
                    'cre_line': cre_line,
                    'structure': structure}]]
    return queries, max([x_width, y_width])


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def flatten_list(l):
    """Flatten list of lists."""
    return [item for sublist in l for item in sublist]


def tweak_params(it_exp):
    """Tweak paramaters to the CC-BP repo."""
    proc_it_exp = {}
    for k, v in it_exp.iteritems():
        if not isinstance(v, list):
            v = [v]
        elif any(isinstance(el, list) for el in v):
            v = flatten_list(v)
        proc_it_exp[k] = v
    return proc_it_exp


def add_experiment(experiment_file, exp_method_template, experiment):
    """Add experiment method to the CC-BP repo."""
    with open(exp_method_template, 'r') as f:
        exp_text = f.readlines()
    for idx, l in enumerate(exp_text):
        exp_text[idx] = exp_text[idx].replace('EDIT', experiment)
        exp_text[idx] = exp_text[idx].replace('RANDALPHA', experiment)
    with open(experiment_file, 'r') as f:
        text = f.readlines()
    text += exp_text
    with open(experiment_file, 'w') as f:
        f.writelines(text)


def get_dt_stamp():
    """Get date-timestamp."""
    return re.split(
        '\.', str(datetime.now()))[0].replace(
        ' ',
        '_').replace(
        ':',
        '_').replace(
        '-',
        '_')


def hp_optim_parameters(parameter_dict, ms_key='model_struct'):
    """Experiment parameters in the case of hp_optimization algorithms."""
    model_structs = parameter_dict[ms_key]
    parameter_dict = {
        k: v for k, v in parameter_dict.iteritems() if k is not ms_key}
    combos = []
    for ms in model_structs:
        it_dict = {}
        for k, v in parameter_dict.iteritems():
            if '_domain' in k:
                if isinstance(v, np.ndarray):
                    v = pd.Series(v).to_json(orient='values')
                elif isinstance(v, basestring):
                    pass
                else:
                    v = json.dumps(v)
            it_dict[k] = v  # Handle special-case hp optim flags here.
        it_dict[ms_key] = ms
        combos += [it_dict]
    return combos


def hp_opt_dict():
    return {
        'regularization_type_domain': 'regularization_type',
        'regularization_strength_domain': 'regularization_strength',
        'optimizer_domain': 'optimizer',
        'lr_domain': 'lr',
        'timesteps_domain': 'timesteps',
        'u_t_domain': 'tuning_u',
        't_t_domain': 'tuning_t',
        'q_t_domain': 'tuning_q',
        'p_t_domain': 'tuning_p',
        'filter_size_domain': 'filter_size',
    }


def prep_exp(
        experiment_dict,
        # db_config,
        credentials,
        cluster=False):
    """Prepare exps for contextual circuit repo."""
    if 'hp_optim' in experiment_dict.keys() and experiment_dict['hp_optim'] is not None:
        exp_combos = hp_optim_parameters(experiment_dict)
        # it_exp = tweak_params(it_exp)
    else:
        exp_combos = package_parameters(experiment_dict)
    with allen_db(
            # config=db_config,
            cluster=cluster,
            credentials=credentials) as db_conn:
        db_conn.populate_db(exp_combos)


def query_hp_hist(
        experiment_name,
        # db_config,
        credentials,
        cluster=False):
    """Get performance from contextual circuit repo."""
    perfs = None
    with allen_db(
            # config=db_config,
            cluster=cluster,
            credentials=credentials) as db_conn:
        perfs = db_conn.get_performance(experiment_name)
    return perfs


def sel_exp_query(
        experiment_name,
        model,
        # db_config,
        credentials,
        cluster=False):
    """Get a select experiment/model combo."""
    perfs = None
    proc_model_name = '%%/%s' % model
    with allen_db(
            # config=db_config,
            cluster=cluster,
            credentials=credentials) as db_conn:
        perfs = db_conn.get_performance_by_model(
            experiment_name=experiment_name,
            model=proc_model_name)
    return perfs


def package_parameters(parameter_dict):
    """Derive combinations of experiment parameters."""
    parameter_dict = {
        k: v for k, v in parameter_dict.iteritems() if isinstance(v, list)
    }
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    combos = tuple({k: v for k, v in zip(keys_sorted, row)} for row in values)
    return list(combos)


def prepare_connection(unpw, port=''):
    """Package DB credentials into a dictionary."""
    params = {
        'database': unpw['database'],
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    return params


class allen_db(object):
    def __init__(
            self,
            # config,
            credentials,
            cluster=False):
        """Init global variables."""
        self.status_message = False
        self.db_schema_file = os.path.join('data_db','db_schema.txt')
        self.cluster = cluster
        # Pass config -> this class
        # self.add_attributes(config)
        self.credentials = credentials

    def add_attributes(self, x):
        """Add attributes to the class."""
        for k, v in x.items():
            setattr(self, k, v)

    def __enter__(self):
        if self.cluster:
            ssh_info = self.credentials.machine_credentials()
            forward = sshtunnel.SSHTunnelForwarder(
                ssh_info['ssh_address'],
                ssh_username=ssh_info['username'],
                ssh_password=ssh_info['password'],
                remote_bind_address=('127.0.0.1', 5432))
            forward.start()
            self.forward = forward
            self.pgsql_port = forward.local_bind_port
            pgsql_string = prepare_connection(
                self.credentials.cluster_cmbp_postgresql_credentials(),
                str(self.pgsql_port))
        else:
            self.forward = None
            self.pgsql_port = ''
            pgsql_string = prepare_connection(
                self.credentials.cmbp_postgresql_credentials(),
                str(self.pgsql_port))
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        if self.forward is not None:
            self.forward.close()
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        return self

    def close_db(self, commit=True):
        """Commit changes and exit the DB."""
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def experiment_fields(self):
        """Dict of fields in exp & hp_combo_history tables. DEPRECIATED."""
        return {
            'experiment_name': ['experiments', 'hp_combo_history'],
            'model_struct': ['experiments', 'hp_combo_history'],
            'loss_function': ['experiments', 'hp_combo_history'],
            'regularization_type': ['experiments', 'hp_combo_history'],
            'regularization_strength': ['experiments', 'hp_combo_history'],
            'optimizer': ['experiments', 'hp_combo_history'],
            'lr': ['experiments', 'hp_combo_history'],
            'dataset': ['experiments', 'hp_combo_history'],
            'regularization_type_domain': ['experiments', 'hp_combo_history'],
            'regularization_strength_domain': [
                'experiments', 'hp_combo_history'],
            'optimizer_domain': ['experiments', 'hp_combo_history'],
            'lr_domain': ['experiments', 'hp_combo_history'],
            'timesteps': ['experiments', 'hp_combo_history'],
            'timesteps_domain': ['experiments', 'hp_combo_history'],
            'filter_size': ['experiments', 'hp_combo_history'],
            'filter_size_domain': ['experiments', 'hp_combo_history'],
            'u_t_domain': ['experiments', 'hp_combo_history'],
            'q_t_domain': ['experiments', 'hp_combo_history'],
            't_t_domain': ['experiments', 'hp_combo_history'],
            'p_t_domain': ['experiments', 'hp_combo_history'],
            'u_t': ['experiments', 'hp_combo_history'],
            'q_t': ['experiments', 'hp_combo_history'],
            't_t': ['experiments', 'hp_combo_history'],
            'p_t': ['experiments', 'hp_combo_history'],
            'hp_optim': ['experiments', 'hp_combo_history'],
            'hp_max_studies': ['experiments', 'hp_combo_history'],
            'hp_current_iteration': ['experiments', 'hp_combo_history'],
            'normalize_labels': ['experiments', 'hp_combo_history'],
            'experiment_iteration': ['experiments', 'hp_combo_history']
        }


    def fix_namedict(self, namedict, table):
        """Insert empty fields in dictionary where keys are absent."""
        experiment_fields = self.experiment_fields()
        for idx, entry in enumerate(namedict):
            for k, v in experiment_fields.iteritems():
                if k == 'experiment_iteration':
                    # Initialize iterations at 0
                    entry[k] = 0
                elif k not in entry.keys():
                    entry[k] = None
            namedict[idx] = entry
        return namedict

    def recreate_db(self, run=False):
        """Initialize the DB from the schema file."""
        if run:
            db_schema = open(self.db_schema_file).read().splitlines()
            for s in db_schema:
                t = s.strip()
                if len(t):
                    self.cur.execute(t)

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                    )

    def populate_db(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        namedict = self.fix_namedict(namedict, 'experiments')
        if not experiment_link:
            self.cur.executemany(
                """
                INSERT INTO experiments
                (
                experiment_name,
                model_struct,
                loss_function,
                regularization_type,
                regularization_strength,
                optimizer,
                lr,
                dataset,
                regularization_type_domain,
                regularization_strength_domain,
                optimizer_domain,
                lr_domain,
                timesteps,
                timesteps_domain,
                u_t_domain,
                q_t_domain,
                t_t_domain,
                p_t_domain,
                u_t,
                q_t,
                t_t,
                p_t,
                hp_optim,
                hp_max_studies,
                hp_current_iteration,
                experiment_iteration,
                normalize_labels,
                filter_size,
                filter_size_domain
                )
                VALUES
                (
                %(experiment_name)s,
                %(model_struct)s,
                %(loss_function)s,
                %(regularization_type)s,
                %(regularization_strength)s,
                %(optimizer)s,
                %(lr)s,
                %(dataset)s,
                %(regularization_type_domain)s,
                %(regularization_strength_domain)s,
                %(optimizer_domain)s,
                %(lr_domain)s,
                %(timesteps)s,
                %(timesteps_domain)s,
                %(u_t_domain)s,
                %(q_t_domain)s,
                %(t_t_domain)s,
                %(p_t_domain)s,
                %(u_t)s,
                %(q_t)s,
                %(t_t)s,
                %(p_t)s,
                %(hp_optim)s,
                %(hp_max_studies)s,
                %(hp_current_iteration)s,
                %(experiment_iteration)s,
                %(normalize_labels)s,
                %(filter_size)s,
                %(filter_size_domain)s
                )
                """,
                namedict)
            self.cur.execute(
                """
                UPDATE experiments
                SET experiment_link=_id
                WHERE experiment_name=%(experiment_name)s
                """,
                namedict[0])
        else:
            self.cur.executemany(
                """
                INSERT INTO experiments
                (
                experiment_name,
                model_struct,
                loss_function,
                regularization_type,
                regularization_strength,
                optimizer,
                lr,
                dataset,
                regularization_type_domain,
                regularization_strength_domain,
                optimizer_domain,
                lr_domain,
                timesteps,
                timesteps_domain,
                u_t_domain,
                q_t_domain,
                t_t_domain,
                p_t_domain,
                u_t,
                q_t,
                t_t,
                p_t,
                hp_optim,
                hp_max_studies,
                hp_current_iteration,
                experiment_iteration,
                normalize_labels,
                filter_size,
                filter_size_domain,
                experiment_link
                )
                VALUES
                (
                %(experiment_name)s,
                %(model_struct)s,
                %(loss_function)s,
                %(regularization_type)s,
                %(regularization_strength)s,
                %(optimizer)s,
                %(lr)s,
                %(dataset)s,
                %(regularization_type_domain)s,
                %(regularization_strength_domain)s,
                %(optimizer_domain)s,
                %(lr_domain)s,
                %(timesteps)s,
                %(timesteps_domain)s,
                %(u_t_domain)s,
                %(q_t_domain)s,
                %(t_t_domain)s,
                %(p_t_domain)s,
                %(u_t)s,
                %(q_t)s,
                %(t_t)s,
                %(p_t)s,
                %(hp_optim)s,
                %(hp_max_studies)s,
                %(hp_current_iteration)s,
                %(experiment_iteration)s,
                %(normalize_labels)s,
                %(filter_size)s,
                %(filter_size_domain)s,
                %(experiment_link)s
                )
                """,
                namedict)
        if self.status_message:
            self.return_status('INSERT')

    def get_parameters(self, experiment_name=None, random=True):
        """Pull parameters DEPRECIATED."""
        if experiment_name is not None:
            exp_string = """experiment_name='%s' and""" % experiment_name
        else:
            exp_string = """"""
        if random:
            rand_string = """ORDER BY random()"""
        else:
            rand_string = """"""
        self.cur.execute(
            """
            SELECT * from experiments h
            WHERE %s NOT EXISTS (
                SELECT 1
                FROM in_process i
                WHERE h._id = i.experiment_id
                )
            %s
            """ % (
                exp_string,
                rand_string
                )
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_parameters_and_reserve(self, experiment_name=None, random=True):
        """Pull parameters and update the in process table."""
        if experiment_name is not None:
            exp_string = """experiment_name='%s' and""" % experiment_name
        else:
            exp_string = """"""
        if random:
            rand_string = """ORDER BY random()"""
        else:
            rand_string = """"""
        self.cur.execute(
            """
            INSERT INTO in_process (experiment_id, experiment_name)
            (SELECT _id, experiment_name FROM experiments h
            WHERE %s NOT EXISTS (
                SELECT 1
                FROM in_process i
                WHERE h._id = i.experiment_id
                )
            %s LIMIT 1)
            RETURNING experiment_id
            """ % (
                exp_string,
                rand_string,
                )
        )
        self.cur.execute(
            """
            SELECT * FROM experiments
            WHERE _id=%(_id)s
            """,
            {
                '_id': self.cur.fetchone()['experiment_id']
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def list_experiments(self):
        """List all experiments."""
        self.cur.execute(
            """
            SELECT distinct(experiment_name) from experiments
            """
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def update_in_process(self, experiment_id, experiment_name):
        """Update the in_process table."""
        self.cur.execute(
            """
             INSERT INTO in_process
             VALUES
             (%(experiment_id)s, %(experiment_name)s)
            """,
            {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('INSERT')

    def get_performance(self, experiment_name):
        """Get experiment performance."""
        self.cur.execute(
            """
            SELECT * FROM performance AS P
            LEFT JOIN experiments ON experiments._id=P.experiment_id
            WHERE P.experiment_name=%(experiment_name)s
            """,
            {
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def get_performance_by_model(self, experiment_name, model):
        """Get experiment performance."""
        self.cur.execute(
            """
            SELECT * FROM performance AS P
            LEFT JOIN experiments ON experiments._id=P.experiment_id
            WHERE P.experiment_name=%(experiment_name)s
            AND model_struct LIKE %(model)s
            """,
            {
                'experiment_name': experiment_name,
                'model': model
            }
        )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchall()

    def remove_experiment(self, experiment_name):
        """Delete an experiment from all tables."""
        self.cur.execute(
            """
            DELETE FROM experiments WHERE experiment_name=%(experiment_name)s;
            DELETE FROM performance WHERE experiment_name=%(experiment_name)s;
            DELETE FROM in_process WHERE experiment_name=%(experiment_name)s;
            """,
            {
                'experiment_name': experiment_name
            }
        )
        if self.status_message:
            self.return_status('DELETE')

    def reset_in_process(self):
        """Reset in process table."""
        self.cur.execute(
            """
            DELETE FROM in_process
            """
        )
        if self.status_message:
            self.return_status('DELETE')

    def update_performance(self, namedict):
        """Update performance in database."""
        self.cur.execute(
            """
            INSERT INTO performance
            (
            experiment_id,
            experiment_name,
            summary_dir,
            ckpt_file,
            training_loss,
            validation_loss,
            time_elapsed,
            training_step
            )
            VALUES
            (
            %(experiment_id)s,
            %(experiment_name)s,
            %(summary_dir)s,
            %(ckpt_file)s,
            %(training_loss)s,
            %(validation_loss)s,
            %(time_elapsed)s,
            %(training_step)s
            )
            RETURNING _id""",
            namedict
            )
        if self.status_message:
            self.return_status('SELECT')


class declare_allen_datasets():
    """Class for declaring datasets to be encoded as tfrecords."""

    def __getitem__(self, name):
        """Method for addressing class methods."""
        return getattr(self, name)

    def __contains__(self, name):
        """Method for checking class contents."""
        return hasattr(self, name)

    def globals(self):
        """Global variables for all datasets."""
        return {
            'tf_types': {  # How to store each in tfrecords
                'neural_trace_trimmed': 'float',
                'proc_stimuli': 'string',
                'ROImask': 'string',
                'pupil_size': 'float',
                'running_speed': 'float',
                'eye_locations_spherical': 'float',
                'cell_specimen_id': 'float',
                'on_center_x': 'float',
                'on_center_y': 'float',
                'off_center_x': 'float',
                'off_center_y': 'float',
                'on_width_x': 'float',
                'off_width_y': 'float',
                'event_index': 'float',
                'stimulus_name': 'string',
                'stimulus_iterations': 'float'
            },
            'include_targets': {  # How to store this data in tfrecords
                # 'neural_trace_trimmed': 'split',
                # 'proc_stimuli': 'split',
                'image': 'split',  # Corresponds to reference_image_key
                'stimulus_name': 'repeat',
                'event_index': 'split',
                'label': 'split',  # Corresponds to reference_label_key
                'ROImask': 'repeat',
                'stimulus_iterations': 'split',
                # 'pupil_size': 'split',
                # 'running_speed': 'split', \
                # 'eye_locations_spherical': 'split',
                'cell_specimen_id': 'repeat',
                # 'on_center_x': 'repeat',
                # 'on_center_y': 'repeat',
                # 'off_center_x': 'repeat',
                # 'off_center_y': 'repeat',
                # 'on_width_x': 'repeat',
                # 'off_width_y': 'repeat'
            },
            'neural_delay': [8, 11],  # MS delay * 30fps for neural data
            'st_conv': False,
            'timecourse': 'final',  # all mean or final; only for st_conv.
            'weight_sharing': True,
            'grid_query': True,  # False = evaluate all neurons at once
            'detrend': False,
            # TODO: Flag for switching between final vs vector neural activity
            'deconv_method': None,
            'randomize_selection': False,
            'warp_stimuli': False,
            'slice_frames': 5,  # None,  # Sample every N frames
            'process_stimuli': {
                    # 'natural_movie_one': {  # 1080, 1920
                    #     'resize': [304, 608],  # [270, 480]
                    #  },
                    # 'natural_movie_two': {
                    #     'resize': [304, 608],  # [270, 480]
                    # },
                    # 'natural_movie_three': {
                    #     'resize': [304, 608],  # [270, 480]
                    # },
                    'natural_scenes': {
                        'pad': [1080, 1920],  # Pad to full movie size
                        'resize': [304, 608],  # [270, 480]
                    },
                },
            # natural_movie_one
            # natural_movie_two
            # natural_movie_three
            # natural_scenes
            'stimuli': [
                'natural_movie_one',
                'natural_movie_two',
                # 'natural_movie_three'
            ],
            'sessions': [
                # 'three_session_A',
                # 'three_session_B',
                'three_session_C',
                'three_session_C2'
            ],
            'data_type': np.float32,
            'image_type': np.float32,
        }

    def template_dataset(self):
        """Pull data from all neurons."""
        exp_dict = {
            'experiment_name': 'ALLEN_all_neurons',
            'only_process_n': None,  # Set to None to process all
            'randomize_selection': True,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_query': [{
                'rf_coordinate_range': {  # Get all cells
                    'x_min': 40,
                    'x_max': 70,
                    'y_min': 20,
                    'y_max': 50,
                },
                'cre_line': 'Cux2',
                'structure': 'VISp',
                'imaging_depth': 175}
            ],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'store_means': [
                'image',
                'label'
            ],
            'cc_repo_vars': {
                'output_size': [2, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            },
            # 'deconv_method': 'elephant'
        }
        exp_dict = self.add_globals(exp_dict)
        return exp_dict

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def ALLEN_ss_cells_1_movies(self):
        """1 cell from across the visual field."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_ss_cells_1_movies'
        exp_dict['only_process_n'] = 1
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': 20,
                'x_max': 30,
                'y_min': 50,
                'y_max': 60,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        # exp_dict['deconv_method'] = 'c2s'
        exp_dict['cc_repo_vars'] = {
                'output_size': [1, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],  # [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        exp_dict['cv_split'] = {
            'cv_split_single_stim': {
                'target': 0,
                'split': 0.9
            }
        }
        # exp_dict['cv_split'] = {
        #         'split_on_stim': 'natural_movie_two'  # Specify train set
        # }
        return exp_dict

    def ALLEN_st_cells_1_movies(self):
        """1 cell from across the visual field."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_st_cells_1_movies'
        exp_dict['only_process_n'] = None  # MICHELE
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': 20,
                'x_max': 30,
                'y_min': 50,
                'y_max': 60,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        # exp_dict['deconv_method'] = 'c2s'
        # exp_dict['cv_split'] = {
        #     'cv_split_single_stim': {
        #         'target': 0,
        #         'split': 0.9
        #     }
        # }
        exp_dict['cv_split'] = {
                'split_on_stim': 'natural_movie_two'  # Specify train set
        }
        exp_dict['neural_delay'] = [8, 13]  # MS delay * 30fps for neural data
        exp_dict['slice_frames'] = 4  # MICHELE
        exp_dict['st_conv'] = len(
            range(
                exp_dict['neural_delay'][0],
                exp_dict['neural_delay'][1]))
        exp_dict['grid_query'] = False  # False = evaluate all neurons at once
        exp_dict['cc_repo_vars'] = {
                'output_size': [1, 1],
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        exp_dict['weight_sharing'] = True
        return exp_dict


def process_dataset(
        dataset_method,
        rf_dict,
        this_dataset_name,
        model_directory,
        model_templates,
        exps,
        template_experiment,
        session_name,
        meta_dir,
        db_config,
        experiment_file,
        main_config,
        N=16,
        idx=0,
        cluster=False,
        filter_size=None,  # NOT IMPLEMENTED YET
        exp_method_template=None):

    # 1. Prepare dataset
    x_min = np.floor(rf_dict['on_center_x'])
    y_min = np.floor(rf_dict['on_center_y'])
    if 'on_center_x_max' in rf_dict:
        x_max = np.floor(rf_dict['on_center_x_max'])
    else:
        x_max = np.floor(rf_dict['on_center_x']) + 1
    if 'on_center_y_max' in rf_dict:
        y_max = np.floor(rf_dict['on_center_y_max'])
    else:
        y_max = np.floor(rf_dict['on_center_y']) + 1
    rf_query = dataset_method['rf_query'][0]
    rf_query['rf_coordinate_range']['x_min'] = x_min
    rf_query['rf_coordinate_range']['x_max'] = x_max
    rf_query['rf_coordinate_range']['y_min'] = y_min
    rf_query['rf_coordinate_range']['y_max'] = y_max
    rf_query['structure'] = rf_dict[
        'structure']
    rf_query['cre_line'] = rf_dict[
        'cre_line']
    rf_query['imaging_depth'] = rf_dict[
        'imaging_depth']
    dataset_method['rf_query'][0] = rf_query
    dataset_name = '%s_%s_%s_%s_%s_%s' % (
        rf_dict['structure'],
        rf_dict['cre_line'],
        rf_dict['imaging_depth'],
        int(rf_query['rf_coordinate_range']['x_min']*100),
        int(rf_query['rf_coordinate_range']['y_min']*100),
        idx)

    print 'Creating dataset %s.' % dataset_name
    method_name = ''.join(
        random.choice(  # TODO: FIX THIS
            string.ascii_uppercase + string.ascii_lowercase)
        for _ in range(N))
    method_name = this_dataset_name + method_name
    dataset_method['experiment_name'] = method_name
    dataset_method['dataset_name'] = dataset_name
    dataset_method['cell_specimen_id'] = rf_dict['cell_specimen_id']
    dataset_method['cc_data_dir'] = main_config.cc_data_dir

    # 2. Encode dataset
    success = encode_datasets.main(dataset_method)

    if success:
        # 3. Prepare models in CC-BP
        new_model_dir = os.path.join(
            model_directory,
            method_name)
        make_dir(new_model_dir)
        for f in model_templates:
            dest = os.path.join(
                new_model_dir,
                f.split(os.path.sep)[-1])
            shutil.copy(f, dest)

        # 4. Add dataset to CC-BP database
        it_exp = exps[template_experiment]()
        it_exp['experiment_name'] = [method_name]  # [dataset_name]
        it_exp['dataset'] = [method_name]  # [dataset_name]
        it_exp['experiment_link'] = [session_name]
        it_exp = tweak_params(it_exp)
        np.savez(
            os.path.join(meta_dir, dataset_name),
            it_exp=it_exp,
            dataset_method=dataset_method,
            rf_data=rf_dict)
        prep_exp(
            it_exp,
            credentials=db_config,
            cluster=cluster)

        # 5. Add the experiment method
        add_experiment(
            experiment_file,
            exp_method_template,
            method_name)


def rf_extents(rf_dict):
    """Find neuron RF extents."""
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    for rf in rf_dict:
        x_min = np.min([rf['on_center_x'], x_min])
        x_max = np.max([rf['on_center_x'], x_max])
        y_min = np.min([rf['on_center_y'], y_min])
        y_max = np.max([rf['on_center_y'], y_max])
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }


def calculate_rf_size(rf_size, downsample):
    """Calculate the appropriate filter size for the input image."""
    h = 61  # 24" monitor
    d = 10  # 10cm from the right eye
    r = 1080 / downsample  # Vertical resolution
    d_px = np.degrees(math.atan2(h / 2, d)) / (r / 2)
    return rf_size * d_px


def build_multiple_datasets(
        template_dataset='ALLEN_st_cells_1_movies',
        template_experiment='ALLEN_st_selected_cells_1',
        model_structs='ALLEN_st_selected_cells_1',
        this_dataset_name='MULTIALLEN_ltwothree_st_',
        cluster=False,
        N=16):
    """Main function for creating multiple datasets of cells."""
    main_config = Allen_Brain_Observatory_Config()

    # Remove any BP-CC repos in the path
    bp_cc_paths = [
        x for x in sys.path if 'contextual' in x or 'Contextual' in x]
    [sys.path.remove(x) for x in bp_cc_paths]

    # Append the BP-CC repo to this python path
    if cluster:
        cc_path = main_config.cluster_cc_path
    else:
        cc_path = main_config.cc_path
    main_config.cc_data_dir = os.path.join(
        cc_path,
        'dataset_processing')  # Pass to encode_datasets.py
    sys.path.append(cc_path)
    import experiments  # from BP-CC
    # from db import credentials
    exps = experiments.experiments()
    # db_config = credentials.postgresql_connection()

    # Query all neurons for an experiment setup
    queries = [  # MICHELE: ADD LOOP HERE
        [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            },
            'cre_line': 'Cux2',  # Scnn1a-Tg3-Cre Nr5a1-Cre # Layer 4 models
            'structure': 'VISp',
            'imaging_depth': 175  # Layer 2/3 models
            }]
    ]
    filter_by_stim = [
        'natural_movie_one',
        'natural_movie_two'
        ]
    sessions = [
        'three_session_C',
        'three_session_C2'
    ]
    print 'Pulling cells by their RFs and stimulus: %s.' % filter_by_stim
    all_data_dicts = query_neurons_rfs(
        queries=queries,
        filter_by_stim=filter_by_stim,
        sessions=sessions)
    # Check if a weight sharing is needed
    dataset_method = declare_allen_datasets()[template_dataset]()
    if dataset_method['weight_sharing']:
        gridded_rfs, rf_size = create_grid_queries(all_data_dicts[0])
        if dataset_method['grid_query']:
            all_data_dicts = query_neurons_rfs(
                queries=gridded_rfs,
                filter_by_stim=filter_by_stim,
                sessions=sessions)
            all_data_dicts = [
                x for x in all_data_dicts if x != []]  # Filter empties.
        downsample = dataset_method[
            'process_stimuli']['natural_scenes']['resize'][0] /\
            dataset_method['cc_repo_vars']['model_im_size'][0]
        filter_size = calculate_rf_size(
            rf_size=rf_size,
            downsample=downsample)
    else:
        filter_size = None

    # Declare the experiment template
    if dataset_method['st_conv']:
        # Dynamic dataset
        exp_method_template = os.path.join(
            main_config.exp_method_template_dir,
            '3d_exp_method_template.txt')
    else:
        # Static dataset
        exp_method_template = os.path.join(
            main_config.exp_method_template_dir,
            '2d_exp_method_template.txt')

    # Prepare directories
    model_directory = os.path.join(
        cc_path,
        'models',
        'structs')
    model_templates = glob(
        os.path.join(
            model_directory,
            model_structs,
            '*.py'))
    experiment_file = os.path.join(cc_path, 'experiments.py')

    # Loop through each query and build all possible datasets with template
    ts = get_dt_stamp()
    session_name = int(''.join(
        [random.choice(string.digits) for k in range(N//2)]))
    for ni, q in enumerate(all_data_dicts):
        assert len(q), 'Cell dictionary is empty.'
        meta_dir = os.path.join(
            main_config.multi_exps,
            '%s_cells_%s' % (len(q), ts))
        make_dir(meta_dir)
        if dataset_method['weight_sharing']:
            print 'Preparing dataset %s/%s.' % (
                ni,
                len(all_data_dicts))
            rf_grid = rf_extents(q)
            rf_dict = q[0]
            rf_dict['on_center_x_max'] = rf_grid['x_max']
            rf_dict['on_center_y_max'] = rf_grid['y_max']
            rf_dict['on_center_x'] = rf_grid['x_min']
            rf_dict['on_center_y'] = rf_grid['y_min']
            process_dataset(
                dataset_method=dataset_method,
                rf_dict=rf_dict,
                this_dataset_name=this_dataset_name,
                model_directory=model_directory,
                model_templates=model_templates,
                exps=exps,
                template_experiment=template_experiment,
                session_name=session_name,
                meta_dir=meta_dir,
                db_config=credentials,  # db_config,
                experiment_file=experiment_file,
                main_config=main_config,
                idx=ni,
                cluster=cluster,
                filter_size=filter_size,
                exp_method_template=exp_method_template)
        else:
            for idx, rf_dict in enumerate(q):
                print 'Preparing dataset %s/%s in package %s/%s.' % (
                    idx,
                    len(q),
                    ni,
                    len(all_data_dicts))
                process_dataset(
                    dataset_method=dataset_method,
                    rf_dict=rf_dict,
                    this_dataset_name=this_dataset_name,
                    model_directory=model_directory,
                    model_templates=model_templates,
                    exps=exps,
                    template_experiment=template_experiment,
                    session_name=session_name,
                    meta_dir=meta_dir,
                    db_config=credentials,  # db_config,
                    experiment_file=experiment_file,
                    main_config=main_config,
                    idx=idx,
                    cluster=cluster,
                    exp_method_template=exp_method_template)
        threading.enumerate()
        import ipdb;ipdb.set_trace()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--cluster',
        dest='cluster',
        action='store_true',
        help='Run experiments on pnodes.')
    args = parser.parse_args()
    build_multiple_datasets(**vars(args))
