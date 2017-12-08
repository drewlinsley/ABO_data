# Config
import os
import socket


class Allen_Brain_Observatory_Config():
    """Config for rebuilding Allen data in a database."""
    def get_host_path(self):
        if socket.gethostname() == 'x8':
            self.host = 'x8'
            self.repo_PATH = '/home/drew/Documents/abo_data'
            self.cc_path = '/home/drew/Documents/contextual_circuit_bp'
        elif socket.gethostname() == 'x9':
            self.host = 'x9'
            self.repo_PATH = '/home/drew/Documents/abo_data'
            # self.cc_path = '/home/drew/Documents/contextual_circuit_bp'
            self.cc_path = '/media/data_cifs/cluster_projects/contextual_circuit_bp'
        else:
            raise Exception(
                'Unknown Host : Please add your directory at get_host_path()')

    def __init__(self, **kwargs):
        self._id = ''
        """ Directories"""
        self.get_host_path()
        self.log_dir = 'logs'
        self.DB_loc = 'DataForTrain'
        self.cc_template = 'template_cc_model.txt'
        self.exp_method_template_dir = 'CCBP_experiment_templates'
        self.tmp_pachaya_folder = 'pachaya_scripts'  # Holding old files here
        self.data_loc = os.path.join(
            '/media',
            'data_cifs',
            'AllenData')
        self.tf_record_output = os.path.join(
            '/media',
            'data_cifs',
            'contextual_circuit',
            'tf_records')
        self.ccbp_exp_evals = os.path.join(
            '/media',
            'data_cifs',
            'contextual_circuit',
            'experiment_evaluations')
        self.deconv_model_dir = os.path.join(
            self.data_loc,
            'deconv_models')
        self.model_struct_dir = os.path.join(
            self.cc_path,
            'models',
            'structs')
        self.model_template_dir = os.path.join(
            self.model_struct_dir,
            'template_ALLEN')
        self.model_prefix = 'auto_generated_'

        # TODO: Document these parameters
        # Parameters
        self.rf_shuffles = 5000
        self.alpha = 0.5
        self.FILTERS = True
        self.filters_file = os.path.join(  # None if no filters
            self.repo_PATH,
            'filters_VISp_175_sigNS_nonzeroNM_reliableRF.pkl')
        self.data_set_code = 'Area-VISp_Depth-175um_NS-sig_NMall-nonzero_LSN-rfChi2-0.05_allStim-true'
        self.cells_pkl = 'all_cells_RF_info_08_30_17.pkl'

        # Order for Name Code
        # Area / depth / Stimuli - SG, DG, NS, NM, LSN / aShow only
        # data with results from all stimuli allStim = True
        self.RESHAPE_IMG_NS = True
        self.reshape_img_size_h = 31
        self.reshape_img_size_w = 31
        self.save_folder = 'DataForTrain/'
        self.db_ssh_forward = False  # True

        # Template for cc_bp repo data loading
        self.multi_exps = 'multi_cell_exps'
        self.cc_data_dir = os.path.join(
            self.cc_path,
            'dataset_processing')
        self.manifest_file = os.path.join(
            self.data_loc,
            'boc/manifest.json')
        self.all_exps_csv = os.path.join(
            self.tmp_pachaya_folder,
            'all_exps.csv')
        self.stimulus_template_loc = os.path.join(
            self.data_loc,
            self.DB_loc,
            'all_stimulus_template')
        self.RF_info_loc = os.path.join(
            self.data_loc,
            self.DB_loc,
            'all_RFs_info')
        self.imaging_response_loc = os.path.join(
            self.data_loc,
            self.DB_loc,
            'all_imaging_responses')
        self.fluoresence_type = 'dff_traces_loc'  # 'fluorescence_traces'
        self.fluorescence_traces_loc = os.path.join(
            self.imaging_response_loc,
            self.fluoresence_type)
        self.ROIs_mask_loc = os.path.join(
            self.imaging_response_loc,
            'ROIs_mask')
        self.stim_table_loc = os.path.join(
            self.imaging_response_loc,
            'stim_tables')
        self.specimen_recording_loc = os.path.join(
            self.imaging_response_loc,
            'specimen_recording')
        self.output_pointer_loc = os.path.join(
            self.data_loc,
            self.DB_loc,
            'output_pointers')
        self.Allen_analysed_stimulus_loc = os.path.join(
            self.data_loc,
            self.DB_loc,
            'Allen_stimulus_analysis',
            'By_cell_ID')
        self.precal_matrix_loc = os.path.join(
            self.data_loc,
            self.DB_loc,
            'Allen_stimulus_analysis',
            'By_container_ID')

        # Brain Observatory project information
        self.stim = {
            'DG': 'drifting_gratings',
            'LSN': 'locally_sparse_noise',
            'LSN4': 'locally_sparse_noise_4deg',
            'LSN8': 'locally_sparse_noise_8deg',
            'NM1': 'natural_movie_one',
            'NM2': 'natural_movie_two',
            'NM3': 'natural_movie_three',
            'NS': 'natural_scenes',
            'Spon': 'spontaneous',
            'SG': 'static_gratings'
        }
        self.session = {
            'A': u'three_session_A',
            'B': u'three_session_B',
            'C': u'three_session_C',
            'C2': u'three_session_C2'
        }
        self.sess_with_number = {
            'locally_sparse_noise_4deg': 'locally_sparse_noise_four_deg',
            'locally_sparse_noise_8deg': 'locally_sparse_noise_eight_deg'}
        self.session_RF_stim = {
            'C': ['locally_sparse_noise'],
            'C2': [
                'locally_sparse_noise_4deg',
                'locally_sparse_noise_8deg'
            ],
        }
        self.session_name_for_RF = [
            'locally_sparse_noise',
            'locally_sparse_noise_4deg',
            'locally_sparse_noise_8deg'
        ]
        self.LSN_size_in_deg = {
            'height': 74.4,
            'width': 130.2
        }
        self.RF_sign = ['on', 'off']
        self.pick_main_RF = [
            'locally_sparse_noise',
            'locally_sparse_noise_8deg'
        ]
        self.available_stims = [
            'locally_sparse_noise',
            'locally_sparse_noise_4deg',
            'locally_sparse_noise_8deg',
            'natural_movie_one',
            'natural_movie_three',
            'natural_movie_two',
            'natural_scenes'
        ]
