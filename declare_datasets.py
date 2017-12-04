"""Declare new allen datasets to be encoded as TFrecords."""
import numpy as np


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
            'neural_delay': [3, 5],  # MS delay * 30fps for neural data
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
            'detrend': True,
            'deconv_method': None,
            'randomize_selection': False,
            'warp_stimuli': False,
            'slice_frames': 15,  # Sample every N frames
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
                'natural_movie_three'
            ],
            'sessions': [
                'three_session_A',
                # 'three_session_B',
                # 'three_session_C'
                # 'three_session_C2'
            ],
            'cv_split': {
                'split_on_stim': 'natural_movie_three'  # Specify train set
            },
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
        exp_dict['cv_split'] = {
            'split_on_stim': 'natural_movie_three'  # Specify train set
        }
        return exp_dict

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    # 10/5/17 datasets
    def ALLEN_selected_cells_1(self):
        """A single cell from the dense RF region."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict = {
            'experiment_name': 'ALLEN_selected_cells_1',
            'only_process_n': 1,  # Set to None to process all
            'randomize_selection': False,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_query': [{
                'rf_coordinate_range': {  # Get all cells
                    'x_min': 40,
                    'x_max': 70,
                    'y_min': 20,
                    'y_max': 50}
            }],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'store_means': [
                'image',
                'label'
            ],
            'cc_repo_vars': {
                'output_size': [1, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            },
            # 'deconv_method': 'elephant'
        }
        exp_dict = self.add_globals(exp_dict)
        return exp_dict

    def ALLEN_selected_cells_1_scenes(self):
        """1 cell from the dense RF region."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_selected_cells_103_scenes'
        exp_dict['only_process_n'] = 1
        exp_dict['randomize_selection'] = False
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': 40,
                'x_max': 70,
                'y_min': 20,
                'y_max': 50,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        exp_dict['cc_repo_vars'] = {
                'output_size': [1, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        exp_dict['sessions'] = [
                'three_session_B',
            ]
        exp_dict['stimuli'] = [
                'natural_scenes'
            ]
        exp_dict['cv_split'] = {
                'split_on_stim': 'natural_scenes'  # Specify train set
            }
        return exp_dict

    def ALLEN_selected_cells_103(self):
        """103 cells from the dense RF region."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_selected_cells_103'
        exp_dict['only_process_n'] = 103
        exp_dict['randomize_selection'] = False
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': 40,
                'x_max': 70,
                'y_min': 20,
                'y_max': 50,
            }}],
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        return exp_dict

    def ALLEN_selected_cells_103_scenes(self):
        """103 cells from the dense RF region."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_selected_cells_103_scenes'
        exp_dict['only_process_n'] = 103
        exp_dict['randomize_selection'] = False
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': 40,
                'x_max': 70,
                'y_min': 20,
                'y_max': 50,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        exp_dict['sessions'] = [
                'three_session_B',
            ]
        exp_dict['stimuli'] = [
                'natural_scenes',
                'natural_movie_one'
            ]
        exp_dict['cv_split'] = {
                'split_on_stim': 'natural_scenes'  # Specify train set
            }
        return exp_dict

    def ALLEN_random_cells_103(self):
        """103 random cells from across the visual field."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_random_cells_103'
        exp_dict['only_process_n'] = 103
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        return exp_dict

    def ALLEN_all_cells(self):
        """All cells from across the visual field that pass filters."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_random_cells_103'
        exp_dict['only_process_n'] = None
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        return exp_dict

    def ALLEN_random_cells_103_scenes(self):
        """103 random cells from across the visual field."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_selected_cells_103_scenes'
        exp_dict['only_process_n'] = 103
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ],
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        exp_dict['sessions'] = [
                'three_session_B',
            ]
        exp_dict['stimuli'] = [
                'natural_scenes',
                'natural_movie_one'
            ]
        exp_dict['cv_split'] = {
                'split_on_stim': 'natural_scenes'  # Specify train set
            }
        return exp_dict

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
                'model_im_size': [152, 304, 1],  # ,
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        return exp_dict

    def ALLEN_ss_cells_max_movies(self):
        """Max cells from a RF location."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_ss_cells_max_movies'
        exp_dict['only_process_n'] = None
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
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [354, 608, 1],  # [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        return exp_dict

    def ALLEN_ns_cells_max_movies(self):
        """103 cells from across the visual field."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_ns_cells_max_movies'
        exp_dict['only_process_n'] = 103
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -1000,
                'x_max': 1000,
                'y_min': -1000,
                'y_max': 1000,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [354, 608, 1],  # [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        exp_dict['cv_split'] = {
                'cv_split': 0.9  # Specify train set
            }
        return exp_dict
