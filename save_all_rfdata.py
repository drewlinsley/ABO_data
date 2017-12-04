"""Compute and save rf data"""
import time
import sys
import glob
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf
from allen_config import Allen_Brain_Observatory_Config
from tqdm import tqdm
import pandas as pd
import numpy as np
from ops import helper_funcs


config = Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)
sys.stdout.flush()


def session_filters(config, cells_ID_list):
    """Find cells that present in all sessions"""
    masterkey = {
        k: v for k, v in config.session.iteritems()
        if v in cells_ID_list.keys()}
    reference_session = masterkey[masterkey.keys()[0]]
    reference_cells = cells_ID_list[reference_session]
    flist = {}  # List of cells that found in all three sessions
    for cell in reference_cells:
        checks = []
        for k in masterkey.keys()[1:]:
            v = cells_ID_list[masterkey[k]]
            checks += [cell in v]
        flist[cell] = all(checks)
    return flist


def filter_cells_for_rf(
        exp_session,
        flist,
        session_RF_stim,
        rf_session=['three_session_C', 'three_session_C2'],
        alpha=0.5,
        number_of_shuffles=5000):
    """Filters for cells that has RFs """
    from helper_funcs import load_object
    save_loc = config.Allen_analysed_stimulus_loc
    rf_cells_dict = {}
    for sess in exp_session:
        it_session = sess['session_type']
        if it_session in rf_session:
            data_set = boc.get_ophys_experiment_data(sess['id'])
            sparse_noise_type = session_RF_stim[it_session.split('_')[-1]]
            for cell_specimen_id, session_filter in tqdm(
                    flist.iteritems(),
                    desc="Deriving RFs for %s" % it_session,
                    total=len(flist)):
                if session_filter:
                    cell_index = data_set.get_cell_specimen_indices(
                        [cell_specimen_id])[0]
                    print("cell %s has index %d\n" % (
                        cell_specimen_id,
                        cell_index))
                    rf_list = []
                    for lsn_deg in sparse_noise_type:
                        if (FORCE_DATA_EXIST):
                            DATA_EXIST = True
                        else:
                            DATA_EXIST = find_rf_data_files(
                                exp_session=exp_session,
                                save_loc=config.Allen_analysed_stimulus_loc)
                        if DATA_EXIST:
                            fname = "%s%s_%s.pkl" % (
                                save_loc,
                                cell_specimen_id,
                                lsn_deg)
                            rf_data = load_object(fname)
                        else:
                            rf_data = rf.compute_receptive_field_with_postprocessing(
                                data_set,
                                cell_index,
                                lsn_deg,
                                alpha=alpha,
                                number_of_shuffles=number_of_shuffles)
                        rf_info = {}
                        rf_info['cell_specimen_id'] = cell_specimen_id
                        rf_info['experiment_container_id'] = sess[
                            'experiment_container_id']
                        rf_info['lsn_name'] = lsn_deg
                        rf_info['alpha'] = alpha
                        rf_info['number_of_shuffles'] = number_of_shuffles
                        rf_info['found_on'] = False
                        rf_info['found_off'] = False
                        for key in config.RF_sign:
                            if 'gaussian_fit' in rf_data[key]:
                                rf_info['found_%s' % key] = True
                                tmp_attr = rf_data[key]['gaussian_fit'][
                                    'attrs']
                                rf_info['%s_distance' % key] = tmp_attr[
                                    'distance']
                                rf_info['%s_area' % key] = tmp_attr[
                                    'area']
                                rf_info['%s_overlap' % key] = tmp_attr[
                                    'overlap']
                                rf_info['%s_height' % key] = tmp_attr[
                                    'height']
                                rf_info['%s_center_x' % key] = tmp_attr[
                                    'center_x']
                                rf_info['%s_center_y' % key] = tmp_attr[
                                    'center_y']
                                rf_info['%s_width_x' % key] = tmp_attr[
                                    'width_x']
                                rf_info['%s_width_y' % key] = tmp_attr[
                                    'width_y']
                                rf_info['%s_rotation' % key] = tmp_attr[
                                    'rotation']
                        rf_list.append(rf_info)
                    rf_cells_dict[cell_specimen_id] = rf_list
    return rf_cells_dict


def get_sess_key(config, sesstxt):
    """Get session key from config."""
    for code, txt in config.session.iteritems():
        if sesstxt == txt:
            return code


def save_all_rf_data(
        exp_session,
        flist,
        session_RF_stim,
        save_loc,
        rf_session=['three_session_C', 'three_session_C2'],
        alpha=0.5,
        number_of_shuffles=5000):
    """Save the stimulus analysis results from Allensdk"""
    from helper_funcs import save_object
    for sess in exp_session:
        it_session = sess['session_type']
        if it_session in rf_session:
            data_set = boc.get_ophys_experiment_data(sess['id'])
            sparse_noise_type = session_RF_stim[it_session.split('_')[-1]]
            for cell_specimen_id, session_filter in tqdm(
                    flist.iteritems(),
                    desc="Deriving RFs for %s" % it_session,
                    total=len(flist)):
                if session_filter:
                    cell_index = data_set.get_cell_specimen_indices(
                        [cell_specimen_id])[0]
                    print(
                        "cell %s has index %d\n" % (
                            cell_specimen_id,
                            cell_index))
                    for lsn_deg in sparse_noise_type:
                        rf_data = rf.compute_receptive_field_with_postprocessing(
                            data_set,
                            cell_index,
                            lsn_deg,
                            alpha=alpha,
                            number_of_shuffles=number_of_shuffles)
                        fname = "%s%s_%s.pkl" % (
                            save_loc,
                            cell_specimen_id,
                            lsn_deg)
                        save_object(rf_data, fname)
                        print("Saved : %s" % (fname))


def find_rf_data_files(exp_session, save_loc):
    """Get file pointers for Allen RF data."""
    rep_sess = exp_session[0]
    tmp_dat = boc.get_ophys_experiment_data(rep_sess['id'])
    cells_id = tmp_dat.get_cell_specimen_ids()
    for cell in cells_id:
        fname = "%s%s_locally_sparse_noise*" % (save_loc, cell)
        file_list = glob.glob(fname)
        if file_list:
            return True
    return False


df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info = []
SAVE_RF_DATA = False
GET_RF_INFO = True
DATE_STAMP = time.strftime('%D').replace('/', '_')
FORCE_DATA_EXIST = True
start = 0
end = len(exp_con_ids)
if len(sys.argv) > 1:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
idx_range = np.arange(start, end)
for idx in tqdm(
        idx_range,
        desc='RF data',
        total=len(idx_range)):
    exps = exp_con_ids[idx]
    print "Running experiment container ID #%s  index = %g of [%g,%g)" % (
        exps,
        idx,
        start,
        end)
    runtime = time.time()
    exp_session = boc.get_ophys_experiments(experiment_container_ids=[exps])
    cells_ID_list = {}
    for sess in exp_session:
        sess_code = get_sess_key(config, sess['session_type'])
        tmp = boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']] = tmp.get_cell_specimen_ids()
    common_cells = session_filters(config, cells_ID_list)

    if SAVE_RF_DATA:
        DATA_EXIST = find_rf_data_files(
            exp_session=exp_session,
            save_loc=config.Allen_analysed_stimulus_loc)
        if DATA_EXIST:
            print "Found files for container ID%s  #%g of [%g,%g)" % (
                exps,
                idx,
                start,
                end)
        else:
            save_all_rf_data(
                exp_session=exp_session,
                flist=common_cells,
                session_RF_stim=config.session_RF_stim,
                save_loc=config.Allen_analysed_stimulus_loc,
                rf_session=['three_session_C', 'three_session_C2'],
                alpha=config.alpha,
                number_of_shuffles=config.rf_shuffles)
    if GET_RF_INFO:
        cells_RF_info = filter_cells_for_rf(
            exp_session=exp_session,
            flist=common_cells,
            session_RF_stim=config.session_RF_stim,
            rf_session=['three_session_C', 'three_session_C2'],
            alpha=config.alpha,
            number_of_shuffles=config.rf_shuffles)
        all_cells_RF_info.append(cells_RF_info)
    print("Run time for experiment container ID #%s is %s " % (
        exps,
        time.time()-runtime))

if GET_RF_INFO:
    helper_funcs.save_object(
        all_cells_RF_info,
        "all_cells_RF_info_%s.pkl" % (DATE_STAMP))
