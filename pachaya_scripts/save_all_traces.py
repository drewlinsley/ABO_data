"""Compute and save recording data"""
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.visualization as rfvis
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf

import matplotlib.pyplot as plt
from config import Allen_Brain_Observatory_Config
from tqdm import tqdm
import pandas as pd
import numpy as np
from db import db
from helper_funcs import *
import time 


config=Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)
import sys
sys.stdout.flush()
from tqdm import tqdm
#python load_all_nwb.py 0 597 2>&1 | tee -a load_log_all.txt

# exp session A : NM1, NM3
# exp session B : NM1, NS
# exp session C : NM1, NM2

def get_sess_key(config, sesstxt):
#    from config import Allen_Brain_Observatory_Config
#    config=Allen_Brain_Observatory_Config()
    for code,txt in config.session.iteritems():
        if sesstxt == txt :
            return code

def session_filters(config,cells_ID_list):
    """Find cells that present in all sessions"""
    masterkey = {k:v for k, v in config.session.iteritems() if v in cells_ID_list.keys()}
    reference_session = masterkey[masterkey.keys()[0]]
    reference_cells = cells_ID_list[reference_session]
    flist = {} #List of cells that found in all three sessions
    for cell in reference_cells:
        checks = []
        for k in masterkey.keys()[1:]:
            v = cells_ID_list[masterkey[k]]
            checks += [cell in v]
        flist[cell] = all(checks)
    return flist

def session_specimen_recording(data_set, sess, save_loc):
    """running speed, motion correction and eye position"""
    from allensdk.brain_observatory.brain_observatory_exceptions import NoEyeTrackingException
    dxcm, dxtime = data_set.get_running_speed()
    running_speed={
    'dxcm':dxcm,
    'dxtime':dxtime}
    try:
        mc = data_set.get_motion_correction()
    except:
        print "Session %s : Error occur in motion_correction"
        mc=None
    timestamps= data_set.get_fluorescence_timestamps()
    fname="%s%s_related_traces.npz"%(save_loc,sess['id'])
    NO_EYE_TRACKING = False
    # Eye Tracking
    try:
        timestamps, locations = data_set.get_pupil_location()
    except NoEyeTrackingException:
        NO_EYE_TRACKING = True
        print("No eye tracking for experiment %s." % data_set.get_metadata()["ophys_experiment_id"])
    if not NO_EYE_TRACKING:
        _, locations_spherical = data_set.get_pupil_location() #(azimuth, altitude)
        _, locations_cartesian = data_set.get_pupil_location(as_spherical=False) #(x,y) in cm
        _, area = data_set.get_pupil_size()
        np.savez(fname, 
            timestamps=timestamps, 
            running_speed=running_speed, 
            motion_correction=mc,
            eye_locations_spherical =locations_spherical,
            eye_locations_cartesian =locations_cartesian,
            pupil_size=area)
    else:
        np.savez(fname, 
            timestamps=timestamps, 
            running_speed=running_speed, 
            motion_correction=mc)
        
    return fname

def save_stim_table(data_set, sess, save_loc): #session - wise
    metadata = sess.copy()
    sess_type=sess['session_type']
    stim_table_loc_pointer={}
    for stim in data_set.list_stimuli():
        stim_table = data_set.get_stimulus_table(stimulus_name=stim)
        fname = "%s%s_%s_%s_table.npz"%(save_loc,sess['id'],sess_type,stim)
        np.savez(fname,
                stim_table=stim_table,
                metadata=metadata)
        key="%s_%s_%s"%(sess['id'], sess_type, stim)
        stim_table_loc_pointer[key] = fname
    return stim_table_loc_pointer 
    
def save_corrected_fluorescence_traces(data_set, common_cell_id, sess, 
                                                            save_loc=config.fluorescence_traces_loc):
    # Save the corrected_fluorescence traces, running speed, motion correction and eye position
    import numpy as np
    sess_type=sess['session_type']
    metadata = sess.copy()
    metadata['common_cell_id']=common_cell_id
    indices=data_set.get_cell_specimen_indices(cell_specimen_ids=common_cell_id)
    sort_id_by_indices=[x for _,x in sorted(zip(indices,common_cell_id))]
    
    metadata['cell_indices']=indices
    try:
        epoch_table = data_set.get_stimulus_epoch_table()
    except:
        print " Error in AllenSDK, No epoch information"
        epoch_table = None

    tstamp, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=sort_id_by_indices) 
    #cell_specimen_id need to be sorted in increasing order by their indices
    traces_loc_pointer = {}
    for i, cid in tqdm(
                    zip(range(len(sort_id_by_indices)),sort_id_by_indices),
                    desc="Saving for %s"%sess['id'],
                    total=len(sort_id_by_indices)):
        corrected_trace=corrected_traces[i,:]        
        fname ="%s%s_%s_traces.npz"%(save_loc,cid,sess_type)
        np.savez(fname,
            tstamp=tstamp, 
            corrected_trace=corrected_trace,
            epoch_table = epoch_table, 
            metadata=metadata)
        key = "%s_%s"%(cid,sess_type)
        traces_loc_pointer[key] = fname
    return traces_loc_pointer

def save_max_projection_forROI(data_set,save_loc):
        max_projection = data_set.get_max_projection() # From imaging / per sess
        fname="%smax_projection_%s.npy"%(save_loc,sess['id'])
        np.save(fname,
        max_projection)
        return fname
def save_ROI_masks(data_set, common_cell_id, sess,  max_projection_fname,
                                                            save_loc=config.ROIs_mask_loc):
    # Save ROI masks
    import numpy as np
    sess_type=sess['session_type']
    metadata = sess.copy()
    metadata['common_cell_id']=common_cell_id
    indices=data_set.get_cell_specimen_indices(cell_specimen_ids=common_cell_id)
    sort_id_by_indices=[x for _,x in sorted(zip(indices,common_cell_id))]
    metadata['cell_indices']=indices
    ROImask_loc_pointer = {}
    for i, cid in tqdm(
                    zip(range(len(sort_id_by_indices)),sort_id_by_indices),
                    desc="Saving for %s"%sess['id'],
                    total=len(sort_id_by_indices)):
        roi_mask = data_set.get_roi_mask(cell_specimen_ids=[cid])[0] #[N, 512,512]
        roi_loc_mask=roi_mask.get_mask_plane()
        roi_shape_mask=roi_mask.mask
        roi_shape_map={
        'x':roi_mask.x, 
        'y':roi_mask.y}
        fname ="%s%s_%s_ROImask.npz"%(save_loc,cid,sess_type)
        np.savez(fname,
            roi_loc_mask=roi_loc_mask,
            roi_shape_mask=roi_shape_mask,
            roi_shape_map=roi_shape_map,
            metadata=metadata,
            max_projection_pointer=max_projection_fname)
        key = "%s_%s"%(cid,sess_type)
        ROImask_loc_pointer[key] = fname
    return ROImask_loc_pointer


#python save_all_traces.py 150 199  2>&1 | tee -a trace_log_150_199.txt
#python save_all_traces.py 56 58  2>&1 | tee -a trace_log_56_58.txt
""" Get RFs information"""
df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info =[]
DATE_STAMP = time.strftime('%D').replace('/','_')
start = 0
end =len(exp_con_ids)
if len(sys.argv) >1:
    start=int(sys.argv[1])
    end=int(sys.argv[2])
idx_range =np.arange(start,end)
#save_check = {}  1) Other recording -- per sess 2) Neural_traces + ROI -- per_cell
#data_per_cess_check[con_id][sessID] T/F 
#data_per_cID_check[con_id][sessID][cID] T/F 
Recorded_specimen_recording_fname={}
Recorded_traces_loc_pointer ={}
Recorded_ROImask_loc_pointer={}
Recorded_stim_table_loc_pointer={}
time_everything=time.time()

for idx in tqdm(
            idx_range,
            desc="Data from the experiment",
            total=len(idx_range)):
    exps=exp_con_ids[idx]
    print "Start running experiment container ID #%s  index = %g of [%g,%g)"%(exps,idx,start,end)
    runtime=time.time()
    #if exps ==560820973:
      #  print "There is bug when getting epoch information : SKIP"
       # continue 
    exp_session = boc.get_ophys_experiments(experiment_container_ids=[exps])
    cells_ID_list={}

    for sess in exp_session:
        sess_code = get_sess_key(config,sess['session_type'])
        tmp=boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']]=tmp.get_cell_specimen_ids()
    common_cells = session_filters(config,cells_ID_list) 
    
    print("Start saving neural traces for cells that presented in all sessions")
    common_cell_id=[]
    for cell_specimen_id, session_filter in common_cells.iteritems():
        if session_filter:
            common_cell_id.append(cell_specimen_id)
    common_cells_info={}
    common_cells_info['cell_specimen_id'] = common_cell_id
    
    for sess in exp_session:
        data_set= boc.get_ophys_experiment_data(sess['id'])
        stim_table_loc_pointer=save_stim_table(data_set, sess, save_loc=config.stim_table_loc)
        Recorded_stim_table_loc_pointer[sess['id']]=stim_table_loc_pointer
        specimen_recording_fname = session_specimen_recording( # Recording from specimen ex. running speed for all cell in this session [session-wise]
                                                                data_set=data_set,
                                                                sess=sess,
                                                                save_loc=config.specimen_recording_loc) 
        Recorded_specimen_recording_fname[sess['id']]=specimen_recording_fname
        traces_loc_pointer= save_corrected_fluorescence_traces(# save neural traces for all cell in this session [cell-wise]
                                                            data_set=data_set,
                                                            common_cell_id=common_cell_id, 
                                                            sess=sess,
                                                            save_loc=config.fluorescence_traces_loc)
        Recorded_traces_loc_pointer[sess['id']]=traces_loc_pointer
        
        max_proj_fname=save_max_projection_forROI(data_set=data_set, save_loc=config.ROIs_mask_loc)
        ROImask_loc_pointer=save_ROI_masks( # save cell mask for all cell in this session [cell-wise]
                            data_set=data_set,
                            common_cell_id=common_cell_id, 
                            sess=sess,
                            max_projection_fname=max_proj_fname,
                            save_loc=config.ROIs_mask_loc)
        Recorded_ROImask_loc_pointer[sess['id']]=ROImask_loc_pointer
    print("Run time for experiment container ID #%s is %s | Time lapse %s"%(exps,time.time()-runtime,time.time()-time_everything))

TIME_STAMP=time.strftime('%H_%M_%S')
save_object(Recorded_specimen_recording_fname,"Recorded_specimen_recording_fname_%s_%s.log"%(DATE_STAMP,TIME_STAMP))
save_object(Recorded_traces_loc_pointer,"Recorded_traces_loc_pointer_%s_%s.log"%(DATE_STAMP,TIME_STAMP))
save_object(Recorded_ROImask_loc_pointer,"Recorded_ROImask_loc_pointer_%s_%s.log"%(DATE_STAMP,TIME_STAMP))
save_object(Recorded_stim_table_loc_pointer,"Recorded_stim_table_loc_pointer_%s_%s.log"%(DATE_STAMP,TIME_STAMP))

print "Total runtime = %s"%(time.time()- time_everything)

""" Note 
Error 


/home/pachaya/miniconda2/envs/tf279/lib/python2.7/site-packages/allensdk/core/brain_observatory_nwb_data_set.pyc in get_stimulus_epoch_table(self)
    183         interval_stimulus_dict = {}
    184         for stimulus in self.list_stimuli():
--> 185             stimulus_interval_list = get_epoch_mask_list(stimulus_table_dict[stimulus], threshold=threshold_dict.get(self.get_session_type(), None))
    186             for stimulus_interval in stimulus_interval_list:
    187                 interval_stimulus_dict[stimulus_interval] = stimulus

/home/pachaya/miniconda2/envs/tf279/lib/python2.7/site-packages/allensdk/core/brain_observatory_nwb_data_set.pyc in get_epoch_mask_list(st, threshold, max_cuts)
     75
     76     if len(cut_inds) > max_cuts:
---> 77         raise Exception('more than 2 epochs cut')
     78
     79     for ii in range(len(cut_inds)+1):

Exception: more than 2 epochs cut



{'acquisition_age_days': 84,
 'cre_line': u'Rbp4-Cre_KL100',
 'donor_name': u'274234',
 'experiment_container_id': 560820973,
 'id': 556363813,
 'imaging_depth': 435,
 'reporter_line': u'Ai93(TITL-GCaMP6f)',
 'session_type': u'three_session_A',
 'specimen_name': u'Rbp4-Cre;Camk2a-tTA;Ai93-274234',
 'targeted_structure': u'VISam'}





"""


# Create DB dict and save output_pointers  at config.output_pointer_loc
"""
            np.savez(
                output_pointer,
                neural_trace, # Neural traces were recorded session-wise --- all stim in this sess point to the same files
                stimuli:numpy_pointer_to_movie_or_images,  # Insert the directory pointer from above (1)
                stim_table:either store the raw stim table or point to it... doesnt matter,
                other_data:such as pupil responses and running)
                
                
    metadata = sess.copy()
    metadata['common_cell_id']=common_cell_id
    metadata['cell_indices']=data_set.get_cell_specimen_indices(cell_specimen_ids=common_cell_id)
    sess_type=sess['session_type']
    stim_table_loc_pointer={}
    
    for stim in data_set.list_stimuli():
        stim_table = data_set.get_stimulus_table(stimulus_name=stim)
        for cell_id in common_cell_id:
            trace_pointer = traces_loc_pointer[cell_id]
            fname = "%s%s_%s_table.npz"%(save_loc,cell_id,sess_type, stim)
            np.savez(fname,
                stim_table=stim_table,
                trace_pointer=trace_pointer, 
                metadata=metadata)
            key="%s_%s_%s"%(cell_id, sess_type, stim)
            stim_table_loc_pointer[key] = fname
    return stim_table_loc_pointer 
"""










"""

rf_flist ={}
for sk, sv in cells_ID_list.iteritems():
    cell_rfs = []  # Boolean
    for cell in sv:
        cell_index = data_set.get_cell_specimen_indices([cell])[0]

    rf_flist[sk] = cell_rfs


cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]

print("cell %d has index %d" % (cell_specimen_id, cell_index))


# (1) loop through all stimuli types and save ONCE into separate numpys

# 
for cell in tqdm(cells, desc='Adding cells to DB', total=len(cells)):
    cell_id  # set cell id number here
    cell_rf_dict = {
        cell_id:cell_id,
        rf_stuff:rf_stuff
    }  # specify RF dict here
    list_of_cell_stim_dicts = []
    for sess in sessions:
        for stim in stimuli:

            output_pointer = os.path.join(output_data_folder, '%s_%s_%s' % (cell_id, session, stimulus)) 
            np.savez(
                output_pointer,
                neural_trace, # Neural traces were recorded session-wise --- all stim in this sess point to the same files
                stimuli:numpy_pointer_to_movie_or_images,  # Insert the directory pointer from above (1)
                stim_table:either store the raw stim table or point to it... doesnt matter,
                other_data:such as pupil responses and running) # also
                
                
            it_stim_dict = {
                cell_id:cell_id,
                cell_npy:output_pointer,
                boolean_for_the_stimulus)
            list_of_cell_stim_dicts += [it_stim_dict]
    db.add_cell_data(
        cell_rf_dict,
        list_of_cell_stim_dicts)


list_of_cell_stim_dicts = [{}]
db.add_cell_data()
"""