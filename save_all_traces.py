"""Compute and save recording data"""
import sys
import time
import pandas as pd
import numpy as np
from ops import helper_funcs
from tqdm import tqdm
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.brain_observatory_exceptions import NoEyeTrackingException
from allen_config import Allen_Brain_Observatory_Config


config = Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)
sys.stdout.flush()


# exp session A : NM1, NM3
# exp session B : NM1, NS
# exp session C : NM1, NM2

def get_sess_key(config, sesstxt):
    for code, txt in config.session.iteritems():
        if sesstxt == txt:
            return code


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


def session_specimen_recording(data_set, sess, save_loc):
    """running speed, motion correction and eye position"""
    dxcm, dxtime = data_set.get_running_speed()
    running_speed = {
        'dxcm': dxcm,
        'dxtime': dxtime
    }
    try:
        mc = data_set.get_motion_correction()
    except:
        print "Session %s : Error occur in motion_correction"
        mc = None
    timestamps = data_set.get_fluorescence_timestamps()
    fname = "%s%s_related_traces.npz" % (save_loc, sess['id'])
    NO_EYE_TRACKING = False
    # Eye Tracking
    try:
        timestamps, locations = data_set.get_pupil_location()
    except NoEyeTrackingException:
        NO_EYE_TRACKING = True
        print "No eye tracking for experiment %s." % data_set.get_metadata()[
            "ophys_experiment_id"]
    if not NO_EYE_TRACKING:
        _, locations_spherical = data_set.get_pupil_location()
        _, locations_cartesian = data_set.get_pupil_location(
            as_spherical=False)
        _, area = data_set.get_pupil_size()
        np.savez(
            fname,
            timestamps=timestamps,
            running_speed=running_speed,
            motion_correction=mc,
            eye_locations_spherical=locations_spherical,
            eye_locations_cartesian=locations_cartesian,
            pupil_size=area)
    else:
        np.savez(
            fname,
            timestamps=timestamps,
            running_speed=running_speed,
            motion_correction=mc)
    return fname


def save_stim_table(data_set, sess, save_loc):
    metadata = sess.copy()
    sess_type = sess['session_type']
    stim_table_loc_pointer = {}
    for stim in data_set.list_stimuli():
        stim_table = data_set.get_stimulus_table(stimulus_name=stim)
        fname = "%s%s_%s_%s_table.npz" % (
            save_loc,
            sess['id'],
            sess_type,
            stim)
        np.savez(
            fname,
            stim_table=stim_table,
            metadata=metadata)
        key = "%s_%s_%s" % (sess['id'], sess_type, stim)
        stim_table_loc_pointer[key] = fname
    return stim_table_loc_pointer


def save_corrected_fluorescence_traces(
        data_set,
        common_cell_id,
        sess,
        save_loc=config.fluorescence_traces_loc):
    # Save the corrected_fluorescence traces, running speed,
    # motion correction and eye position
    sess_type = sess['session_type']
    metadata = sess.copy()
    metadata['common_cell_id'] = common_cell_id
    indices = data_set.get_cell_specimen_indices(
        cell_specimen_ids=common_cell_id)
    sort_id_by_indices = [x for _, x in sorted(zip(indices, common_cell_id))]
    metadata['cell_indices'] = indices

    try:
        epoch_table = data_set.get_stimulus_epoch_table()
    except:
        print " Error in AllenSDK, No epoch information"
        epoch_table = None
    tstamp, corrected_traces = data_set.get_corrected_fluorescence_traces(
        cell_specimen_ids=sort_id_by_indices)

    # cell_specimen_id need to be sorted in increasing order by their indices
    traces_loc_pointer = {}
    for i, cid in tqdm(
                    zip(range(len(sort_id_by_indices)),
                        sort_id_by_indices),
                    desc="Saving for %s" % sess['id'],
                    total=len(sort_id_by_indices)):
        corrected_trace = corrected_traces[i, :]
        fname = "%s%s_%s_traces.npz" % (save_loc, cid, sess_type)
        np.savez(
            fname,
            tstamp=tstamp,
            corrected_trace=corrected_trace,
            epoch_table=epoch_table,
            metadata=metadata)
        key = "%s_%s" % (cid, sess_type)
        traces_loc_pointer[key] = fname
    return traces_loc_pointer


def save_max_projection_forROI(data_set, save_loc):
        max_projection = data_set.get_max_projection()
        fname = "%smax_projection_%s.npy" % (save_loc, sess['id'])
        np.save(
            fname,
            max_projection)
        return fname


def save_ROI_masks(
        data_set,
        common_cell_id,
        sess,
        max_projection_fname,
        save_loc=config.ROIs_mask_loc):
    # Save ROI masks
    import numpy as np
    sess_type = sess['session_type']
    metadata = sess.copy()
    metadata['common_cell_id'] = common_cell_id
    indices = data_set.get_cell_specimen_indices(
        cell_specimen_ids=common_cell_id)
    sort_id_by_indices = [x for _, x in sorted(zip(indices, common_cell_id))]
    metadata['cell_indices'] = indices
    ROImask_loc_pointer = {}
    for i, cid in tqdm(
            zip(range(len(sort_id_by_indices)), sort_id_by_indices),
            desc="Saving for %s" % sess['id'],
            total=len(sort_id_by_indices)):
        roi_mask = data_set.get_roi_mask(cell_specimen_ids=[cid])[0]
        roi_loc_mask = roi_mask.get_mask_plane()
        roi_shape_mask = roi_mask.mask
        roi_shape_map = {
            'x': roi_mask.x,
            'y': roi_mask.y
            }
        fname = "%s%s_%s_ROImask.npz" % (save_loc, cid, sess_type)
        np.savez(
            fname,
            roi_loc_mask=roi_loc_mask,
            roi_shape_mask=roi_shape_mask,
            roi_shape_map=roi_shape_map,
            metadata=metadata,
            max_projection_pointer=max_projection_fname)
        key = "%s_%s" % (cid, sess_type)
        ROImask_loc_pointer[key] = fname
    return ROImask_loc_pointer


""" Get RFs information"""
df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info = []
DATE_STAMP = time.strftime('%D').replace('/', '_')
start = 0
end = len(exp_con_ids)
if len(sys.argv) > 1:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
idx_range = np.arange(start, end)
Recorded_specimen_recording_fname = {}
Recorded_traces_loc_pointer = {}
Recorded_ROImask_loc_pointer = {}
Recorded_stim_table_loc_pointer = {}
time_everything = time.time()

for idx in tqdm(
        idx_range,
        desc="Data from the experiment",
        total=len(idx_range)):
    exps = exp_con_ids[idx]
    print "Running experiment container ID #%s  index = %g of [%g,%g)" % (
        exps, idx, start, end)
    runtime = time.time()
    exp_session = boc.get_ophys_experiments(experiment_container_ids=[exps])
    cells_ID_list = {}

    for sess in exp_session:
        sess_code = helper_funcs.get_sess_key(config, sess['session_type'])
        tmp = boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']] = tmp.get_cell_specimen_ids()
    common_cells = session_filters(config, cells_ID_list)

    print(
        "Start saving neural traces for cells that presented in all sessions")
    common_cell_id = []
    for cell_specimen_id, session_filter in common_cells.iteritems():
        if session_filter:
            common_cell_id.append(cell_specimen_id)
    common_cells_info = {}
    common_cells_info['cell_specimen_id'] = common_cell_id

    for sess in exp_session:
        data_set = boc.get_ophys_experiment_data(sess['id'])
        stim_table_loc_pointer = save_stim_table(
            data_set, sess, save_loc=config.stim_table_loc)
        Recorded_stim_table_loc_pointer[sess['id']] = stim_table_loc_pointer
        # Recording from specimen ex. running speed for all cell in
        # this session [session-wise]
        specimen_recording_fname = session_specimen_recording(
            data_set=data_set,
            sess=sess,
            save_loc=config.specimen_recording_loc)
        Recorded_specimen_recording_fname[
            sess['id']] = specimen_recording_fname
        # save neural traces for all cell in this session [cell-wise]
        traces_loc_pointer = save_corrected_fluorescence_traces(
            data_set=data_set,
            common_cell_id=common_cell_id,
            sess=sess,
            save_loc=config.fluorescence_traces_loc)
        Recorded_traces_loc_pointer[sess['id']] = traces_loc_pointer
        max_proj_fname = save_max_projection_forROI(
            data_set=data_set,
            save_loc=config.ROIs_mask_loc)

        # save cell mask for all cell in this session [cell-wise]
        ROImask_loc_pointer = save_ROI_masks(
            data_set=data_set,
            common_cell_id=common_cell_id,
            sess=sess,
            max_projection_fname=max_proj_fname,
            save_loc=config.ROIs_mask_loc)
        Recorded_ROImask_loc_pointer[sess['id']] = ROImask_loc_pointer
    print(
        "Run time for experiment container ID #%s is %s | Time lapse %s" % (
            exps,
            time.time()-runtime,
            time.time()-time_everything))

TIME_STAMP = time.strftime('%H_%M_%S')
helper_funcs.save_object(
    Recorded_specimen_recording_fname,
    "Recorded_specimen_recording_fname_%s_%s.log" % (DATE_STAMP, TIME_STAMP))
helper_funcs.save_object(
    Recorded_traces_loc_pointer,
    "Recorded_traces_loc_pointer_%s_%s.log" % (DATE_STAMP, TIME_STAMP))
helper_funcs.save_object(
    Recorded_ROImask_loc_pointer,
    "Recorded_ROImask_loc_pointer_%s_%s.log" % (DATE_STAMP, TIME_STAMP))
helper_funcs.save_object(
    Recorded_stim_table_loc_pointer,
    "Recorded_stim_table_loc_pointer_%s_%s.log" % (DATE_STAMP, TIME_STAMP))

print "Total runtime = %s" % (time.time() - time_everything)
