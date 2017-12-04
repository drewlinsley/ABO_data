"""Functions for encoding cells as TFrecords for contextual circuit bp."""

import os
import gc
import cv2
import argparse
import numpy as np
import tensorflow as tf
import cPickle as pickle
from data_db import data_db
from glob import glob
from tqdm import tqdm
from scipy import stats, misc
from sklearn import linear_model
from declare_datasets import declare_allen_datasets as dad
from allen_config import Allen_Brain_Observatory_Config as Config
from allensdk.brain_observatory import stimulus_info
try:
    from ops import helper_funcs, deconvolve
except:
    'Print failed loading Allen utils. Are you doing something fancy?'
# from memory_profiler import profile


def get_field(d, field, alt):
    """Error handling for absent dict keys."""
    if field in d:
        return d[field]
    else:
        return alt


def create_data_loader_class(template_file, meta_dict, output_file):
    """Write a data loader python class for this dataset."""
    with open(template_file, 'r') as f:
        text = f.readlines()
    for k, v in meta_dict.iteritems():
        for idx, t in enumerate(text):
            text[idx] = t.replace(k, str(v))
    with open(output_file, 'w') as f:
        f.writelines(text)


def create_model_files(
        output_size,
        set_name,
        rf_info,
        template_directory,
        model_dir):
    """Generate CC_BP model files for this dataset."""
    model_files = glob(
        os.path.join(
            template_directory,
            '*.py'))
    h = rf_info['h']
    w = rf_info['w']
    k = rf_info['k']
    output_directory = os.path.join(
        model_dir,
        set_name)
    helper_funcs.make_dir(output_directory)
    for mf in model_files:
        with open(mf) as f:
            tf = f.readlines()
        for idx, l in enumerate(tf):
            tf[idx] = l.replace('OUTPUT_SIZE', str(output_size))
            tf[idx] = l.replace('H_PIX', str(h))
            tf[idx] = l.replace('W_PIX', str(w))
            tf[idx] = l.replace('SIGMA', str(k))
        output_file = os.path.join(
            output_directory,
            mf.split('/')[-1])
        with open(output_file, 'w') as f:
            f.writelines(tf)
        print 'Created model file: %s' % output_file


def fixed_len_feature(length=[], dtype='int64'):
    """Return variables for loading data with tensorflow."""
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)
    elif dtype == 'float':
        return tf.FixedLenFeature(length, tf.float32)
    else:
        raise RuntimeError('Cannot understand the fixed_len_feature dtype.')


def load_data(f, allow_pkls=False):
    """Load data from npy or pickle files."""
    f = f.strip('"')
    ext = f.split('.')[-1]
    assert ext is not None, 'Cannot find an extension on file: %s' % f
    if not allow_pkls:
        ext = 'npy'
        f = '%s.npy' % f.split('.')[0]
    if ext == 'npy' or ext == 'npz':
        return np.load(f)
    elif ext == 'pkl' or ext == 'p':
        out = pickle.load(open(f, 'rb'))
        return out
    else:
        raise RuntimeError('Cannot understand file extension %s' % ext)


def bytes_feature(values):
    """Encode an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Encode an int list into a tf int64 list for a tfrecord."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """TF floating point feature for tfrecords."""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fix_malformed_pointers(d, filter_ext='.npy', rep_ext='.npz'):
    """Replace .npy extensions if they were erroniously placed in the DB."""
    if '.npy' in d:
        d = '%s%s' % (d.split(filter_ext)[0], rep_ext)
    return d


# @profile
def process_body(
        d,
        exp_dict,
        stimuli_key,
        neural_key,
        deconv,
        all_stimuli=None):
    """Process cell data body function."""
    df = {}
    cell_data = load_cell_meta(d)
    cell_id = d['cell_specimen_id']
    df['cell_specimen_id'] = cell_id

    # Stim table
    stim_table = load_data(
        cell_data['stim_table'].item(),
        allow_pkls=True)
    stim_table = stim_table['stim_table']

    # Stimuli
    # raw_stimuli = all_stimuli[cell_data['stim_template'].item()]['raw']
    df['stimulus_name'] = cell_data['stim_template'].item()
    # df['raw_stimuli'] = raw_stimuli
    proc_stimuli = all_stimuli[cell_data['stim_template'].item()]['processed']
    df['proc_stimuli'] = proc_stimuli

    # Neural data
    neural_data = load_data(
        cell_data['neural_trace'].item(),
        allow_pkls=True)
    # TODO: Register fields in pachaya's data creation with create_db.py
    # to avoid the below.
    trace_key = [k for k in neural_data.keys() if 'trace' in k]
    if trace_key is None:
        raise RuntimeError(
            'Could not find a \'trace\' key in the neural_data dict.' +
            'Found the following keys: %s' % neural_data.keys())
    neural_data = neural_data[trace_key[0]].astype(exp_dict['data_type'])
    df['neural_trace'] = neural_data

    # Trim neural data
    if exp_dict['deconv_method'] is not None:
        neural_data = deconv.deconvolve(
            neural_data).astype(exp_dict['data_type'])

    # Delay data with 'neural_delay'
    if isinstance(exp_dict['neural_delay'], list):
        # Average events
        slice_inds = range(
            exp_dict['neural_delay'][0],
            exp_dict['neural_delay'][1])
        neural_data_trimmed = np.zeros((
            len(slice_inds),
            len(stim_table[:, 1])))
        print 'Averaging signal with %s events.' % len(slice_inds)
        for idx in range(len(slice_inds)):
            it_stim_table_idx = stim_table[:, 1] + slice_inds[idx]
            neural_data_trimmed[idx] = neural_data[it_stim_table_idx]
        # Use the first event in stim_table as the triggering event
        stim_table_idx = stim_table[:, 1] + slice_inds[0]
        neural_data_trimmed = neural_data_trimmed.mean(0)
    else:
        # Constant offset
        print 'Using constant offset of %s events.' % exp_dict['neural_delay']
        stim_table_idx = stim_table[:, 1] + exp_dict['neural_delay']
        neural_data_trimmed = neural_data[stim_table_idx]
    if exp_dict['detrend']:
        regr = linear_model.LinearRegression()
        model = regr.fit(
            np.arange(len(neural_data_trimmed)).reshape(-1, 1),
            neural_data_trimmed)
        neural_data_trimmed = neural_data_trimmed.reshape(
            -1) - model.predict(
                neural_data_trimmed.reshape(-1, 1)).reshape(-1)
        neural_data_trimmed = neural_data_trimmed.reshape(-1)
    df['neural_trace_trimmed'] = neural_data_trimmed

    # ROI mask
    roi_mask = load_data(
        cell_data['ROImask'].item(),
        allow_pkls=True)
    roi_mask = roi_mask['roi_loc_mask']
    df['ROImask'] = np.expand_dims(
        roi_mask, axis=-1).astype(exp_dict['image_type'])

    # AUX data
    aux_data = load_data(
        cell_data['other_recording'].item(),
        allow_pkls=True)

    pupil_size = get_field(aux_data, 'pupil_size', None)
    running_speed = get_field(aux_data, 'running_speed', None)
    eye_locations_spherical = get_field(
        aux_data,
        'eye_locations_spherical',
        None)
    eye_locations_cartesian = get_field(
        aux_data,
        'eye_locations_cartesian',
        None)
    if pupil_size is not None:
        df['pupil_size'] = pupil_size[stim_table_idx]
    else:
        df['pupil_size'] = None
    if running_speed is not None:
        df['running_speed'] = running_speed.item()['dxcm'][stim_table_idx]
    else:
        df['running_speed'] = None
    if eye_locations_spherical is not None:
        df['eye_locations_spherical'] = eye_locations_spherical[
            stim_table_idx, :]
    else:
        df['eye_locations_spherical'] = None
    if eye_locations_cartesian is not None:
        df['eye_locations_cartesian'] = eye_locations_cartesian[
            stim_table_idx, :]
    else:
        df['eye_locations_cartesian'] = None

    # RF data
    df['on_center_x'] = d['on_center_x']
    df['on_center_y'] = d['on_center_y']
    df['off_center_x'] = d['off_center_x']
    df['off_center_y'] = d['off_center_y']
    df['on_height'] = d['on_height']
    df['on_width'] = d['on_width_x']
    df['off_height'] = d['off_height']
    df['off_width'] = d['off_width_x']

    # Rename images/neural traces for contextual_circuit_bp
    df['image'] = df.pop(stimuli_key.keys()[0])
    df['label'] = df.pop(neural_key.keys()[0])
    df['event_index'] = range(len(df['label']))

    # Package data
    df = {
        k: v for k, v in df.iteritems()
        if k in exp_dict['include_targets']
    }
    return df


def load_cell_meta(d):
    """Wrapper and error handeling for loading cell meta data."""
    data_pointer = fix_malformed_pointers(d['cell_output_npy'])
    try:
        cell_data = load_data(data_pointer, allow_pkls=True)
    except:
        print 'WARNING: Fix the npz extensions for %s' % data_pointer
        cell_data = load_data(
            '%s.npy.npz' % data_pointer.strip('.npz'), allow_pkls=True)
    return cell_data


def get_stim_names_and_orders(data_dicts):
    """Get the order and name of stimuli."""
    stim_names, stim_orders = [], []
    # Get a list of all stimuli
    for d in tqdm(
            data_dicts,
            total=len(data_dicts),
            desc='Loading stimulus names and orders'):
        cell_data = load_cell_meta(d)
        stim_names += [cell_data['stim_template'].item()]
        stim_table = load_data(
            cell_data['stim_table'].item(),
            allow_pkls=True)
        stim_orders += [stim_table['stim_table'][:, 0]]
    return stim_orders, stim_names


def preload_raw_stimuli(data_dicts, exp_dict):
    """Preload all stimuli to save memory."""
    stim_orders, stim_names = get_stim_names_and_orders(data_dicts)
    np_stim_names = np.asarray(stim_names)

    # Having filtered stimuli by order, slice by the first saved order.
    unique_stimuli = np.unique(stim_names)
    unique_orders = {}
    for stim in unique_stimuli:
        stim_idxs = np_stim_names == stim
        it_orders = np.asarray(
            [x for x, check in zip(stim_orders, stim_idxs) if check])
        unique_orders[stim] = it_orders[0]

    # Apply warping to stimuli if requested
    if exp_dict['warp_stimuli']:
        print 'Warping stimuli.'
        warpers = {}
        panel_size = stimulus_info.MONITOR_DIMENSIONS
        spatial_unit = 'cm'
        origin = 'upper'
        for stim in unique_stimuli:
            if 'movie' in stim:
                n_pixels_r, n_pixels_c = stimulus_info.NATURAL_MOVIE_PIXELS
                warpers[stim] = stimulus_info.Monitor(
                    n_pixels_r=n_pixels_r,
                    n_pixels_c=n_pixels_c,
                    panel_size=panel_size,
                    spatial_unit=spatial_unit).natural_movie_image_to_screen
            elif 'scene' in stim:
                n_pixels_r, n_pixels_c = stimulus_info.NATURAL_SCENES_PIXELS
                warpers[stim] = stimulus_info.Monitor(
                    n_pixels_r=n_pixels_r,
                    n_pixels_c=n_pixels_c,
                    panel_size=panel_size,
                    spatial_unit=spatial_unit).natural_scene_image_to_screen
            else:
                raise RuntimeError('Warping not implemented for this stim.')

    # Load the stimuli into memory
    all_stimuli = {}
    for stim in tqdm(
            unique_stimuli,
            total=len(unique_stimuli),
            desc='Loading stimuli into memory'):
        raw_stimuli = np.load(stim).astype(exp_dict['image_type'])
        if exp_dict['warp_stimuli']:
            # Apply warping to stimuli
            print 'Warping %s...' % stim
            raw_stimuli = np.asarray([warpers[stim](
                img=im,
                origin=origin) for im in raw_stimuli])
        process = [
            v for k, v in exp_dict['process_stimuli'].iteritems()
            if k in stim]
        if len(process):
            process_dict = process[0]
            if 'crop' in process_dict.keys():
                raise RuntimeError('Cropping not implemented.')
            if 'pad' in process_dict.keys():
                pad = process_dict['pad']
                print 'Padding %s to %s...' % (stim, pad)
                im_size = raw_stimuli[0].shape[:2]
                pad_to = np.asarray(pad) - im_size
                pad_to = pad_to // 2
                raw_stimuli = [cv2.copyMakeBorder(
                    im,
                    top=pad_to[0],
                    bottom=pad_to[0],
                    left=pad_to[1],
                    right=pad_to[1],
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]) for im in raw_stimuli]
            if 'resize' in process_dict.keys():
                resize = process_dict['resize']
                print 'Resizing %s to %s...' % (stim, resize)
                raw_stimuli = np.asarray(
                    [misc.imresize(im, resize) for im in raw_stimuli])
        if len(raw_stimuli.shape) < 4:
            # Ensure that stimuli are a 4D tensor.
            raw_stimuli = np.expand_dims(raw_stimuli, axis=-1)
        all_stimuli[stim] = {
            # 'raw': raw_stimuli,
            'processed': raw_stimuli[unique_orders[stim]]
        }
    return all_stimuli


def process_cell_data(
        data_dicts,
        exp_dict,
        stimuli_key,
        neural_key):
    """Loop for processing cell data."""
    deconv = deconvolve.deconvolve(exp_dict)

    # Preprocess raw_stimuli
    all_stimuli = preload_raw_stimuli(data_dicts=data_dicts, exp_dict=exp_dict)

    # Variables
    key_list = []
    output_data = []
    for d in tqdm(data_dicts, total=len(data_dicts), desc='Preparing data'):
        df = process_body(
            d=d,
            exp_dict=exp_dict,
            stimuli_key=stimuli_key,
            neural_key=neural_key,
            deconv=deconv,
            all_stimuli=all_stimuli)
        output_data += [df]
        it_check = [k for k, v in df.iteritems() if v is not None]
        key_list += [it_check]
        del df
        gc.collect()
    return output_data, key_list


def load_npzs(
        data_dicts,
        exp_dict,
        stimuli_key=None,
        neural_key=None,
        check_stimuli=False):
    """Load cell data from an npz."""

    # Organize data_dicts by cell
    cell_specimen_ids = [d['cell_specimen_id'] for d in data_dicts]
    unique_cells = np.unique(cell_specimen_ids)

    # Trim to specified number of cells if desired.
    exp_dict['only_process_n'] = np.min(
        [
            len(unique_cells),
            exp_dict['only_process_n']])
    if exp_dict[
            'only_process_n'] is not None or exp_dict['only_process_n'] > 0:
        print 'Trimming query from %s to %s cells.' % (
            len(unique_cells), exp_dict['only_process_n'])
        if exp_dict['randomize_selection']:
            unique_cells = unique_cells[np.random.permutation(
                len(unique_cells))]

        if exp_dict['sessions'] is not None:
            # Filter cells that have entries in specified sessions.
            session_dict = {}
            for d in data_dicts:
                cell = d['cell_specimen_id']
                if cell not in session_dict.keys():
                    session_dict[cell] = []
                session_dict[cell] += [d['session']]
            print 'Sorting cells by sessions: %s' % exp_dict[
                'sessions']
            session_counts = {k: len(v) for k, v in session_dict.iteritems()}
            unique_cells = np.asarray(session_counts.keys())[np.argsort(
                session_counts.values())[::-1]]
        keep_cells = unique_cells[:exp_dict['only_process_n']]
        data_dicts = [
            d for d in data_dicts if d['cell_specimen_id'] in keep_cells]

    # Main data processing loop
    output_data, key_list = process_cell_data(
        data_dicts,
        exp_dict,
        stimuli_key,
        neural_key)

    if check_stimuli:
        # Pause to inspect the remaining stimuli
        stim_check = np.unique([d['stimulus_name'] for d in output_data])
        print '-' * 20
        print 'Remaining stimuli are: %s.' % stim_check
        print '-' * 20
        choice = raw_input('Would you like to continue? (y/n)')
        if choice == 'n' or choice == 'N':
            print 'Exiting...'
            os._exit(1)

    keep_keys = np.unique(key_list)
    remove_keys = list(
        set(exp_dict['include_targets'].keys()) - set(keep_keys))
    if remove_keys is not None:
        print 'Removing keys that were not populated across cells: %s.' %\
            remove_keys
        for idx, d in enumerate(output_data):
            it_d = {k: v for k, v in d.iteritems() if k in keep_keys}
            output_data[idx] = it_d

    # TODO: handle NaNs in output_data here.
    if exp_dict['cc_repo_vars']['output_size'][0] > 0:  # 1:
        # Multi neuron target; consolidate event_dict.
        stimuli = [d['stimulus_name'] for d in output_data]
        unique_stimuli, ri = np.unique(
            stimuli,
            return_index=True)

        # Concatenate image/labels appropriately.
        labels = {}
        ROImasks = {}
        images = {}
        cell_specimen_ids = {}
        for stim in zip(unique_stimuli):
            stim = stim[0]
            labels[stim] = []
            ROImasks[stim] = []
            images[stim] = []
            cell_specimen_ids[stim] = []
            for d, rd in zip(output_data, data_dicts):
                if d['stimulus_name'] == stim:
                    labels[stim] += [d['label'][:, None]]
                    ROImasks[stim] += [np.expand_dims(d['ROImask'], axis=0)]
                    images[stim] += [d['image']]
                    cell_specimen_ids[stim] += [d['cell_specimen_id']]

        # Process dicts for cells
        cat_labels = {}
        cat_ROImasks = {}
        cat_images = {}
        cat_repeats = {}
        cat_events = {}
        cat_cell_specimen_ids = {}
        for stim in unique_stimuli:
            cells = np.asarray(cell_specimen_ids[stim])
            unique_cells = np.unique(cells)
            cat_cell_specimen_ids[stim] = unique_cells

            # Count the floor number of times each cell was recorded
            count_cells = cells - cells.min()
            cell_bins = np.bincount(count_cells)
            cell_floor = cell_bins[cell_bins > 0].min()

            # Concatenate cells up to cell_floor times
            for cell_count, cell in enumerate(unique_cells):
                cell_ids = np.where(cells == cell)[0][:cell_floor]
                for cell_it, ci in enumerate(cell_ids):
                    if cell_it == 0:
                        cell_labels = labels[stim][ci]
                        cell_images = images[stim][ci]
                        cell_ROImasks = ROImasks[stim][ci]
                        cell_events = np.arange(len(cell_labels))
                        cell_repeats = np.ones(len(cell_labels)) * cell_it
                    else:
                        cell_labels = np.concatenate(
                            (
                                cell_labels,
                                labels[stim][ci]),
                            axis=0)
                        cell_images = np.concatenate(
                            (
                                cell_images,
                                images[stim][ci]),
                            axis=0)
                        cell_events = np.concatenate(
                            (
                                cell_events,
                                np.arange(len(cell_labels))),
                            axis=0)
                        cell_repeats = np.concatenate(
                            (
                                cell_repeats,
                                np.ones(len(cell_labels)) * cell_it),
                            axis=0)
                if cell_count == 0:
                    cat_labels[stim] = cell_labels
                    cat_ROImasks[stim] = cell_ROImasks
                    cat_images[stim] = cell_images
                    cat_events[stim] = cell_events
                    cat_repeats[stim] = cell_repeats
                else:
                    # Concatenate cells across dimensions
                    cat_labels[stim] = np.concatenate(
                        (
                            cat_labels[stim],
                            cell_labels
                        ),
                        axis=1)

                    # Masks are inconsistently sized
                    pad_offset = np.abs(
                        np.asarray(
                            cat_ROImasks[stim].shape[1:3]) - np.asarray(
                            cell_ROImasks.shape[1:3]))

                    if np.any(pad_offset):
                        # Add padding -- This isn't correctly aligning cells.
                        cell_ROImasks = cv2.copyMakeBorder(
                            cell_ROImasks.squeeze(),
                            pad_offset[0],
                            0,
                            pad_offset[1],
                            0,
                            cv2.BORDER_CONSTANT,
                            0)[None, :, :, None]
                    cat_ROImasks[stim] = np.concatenate(
                        (
                            cat_ROImasks[stim],
                            cell_ROImasks
                        ),
                        axis=0)

        # Test for aligned cells across sessions
        # import ipdb;ipdb.set_trace()  TODO: FIX THIS FOR SCENES
        test_cells = np.concatenate(
            [np.expand_dims(x, axis=-1)
                for x in cat_cell_specimen_ids.values()],
            axis=-1)
        assert test_cells.var(-1).sum() == 0, 'Cell IDs are not aligned.'

        # Prepare meta rf dict
        cell_list = test_cells[:, 0]
        output_rfs = {}
        for ce in cell_list:
            dict_list = []
            for d in data_dicts:
                if d['cell_specimen_id'] == ce:
                    dict_list += [d]
            output_rfs[ce] = dict_list

        # Package into a list of dicts.
        output_data = []
        for (ik, iv), (lk, lv), (rk, rv), (ck, cv), (pk, pv), (ek, ev) in zip(
                cat_images.iteritems(),
                cat_labels.iteritems(),
                cat_ROImasks.iteritems(),
                cat_cell_specimen_ids.iteritems(),
                cat_repeats.iteritems(),
                cat_events.iteritems()):
            assert ik == lk == rk == ck == pk == ek, 'Issue with keys.'
            if exp_dict['slice_frames'] is not None:
                iv = iv[range(0, len(iv), exp_dict['slice_frames'])]
                lv = lv[range(0, len(lv), exp_dict['slice_frames'])]
                rv = rv[range(0, len(rv), exp_dict['slice_frames'])]
                pv = pv[range(0, len(pv), exp_dict['slice_frames'])]
                ev = ev[range(0, len(ev), exp_dict['slice_frames'])]
            output_data += [{
                'image': iv,
                'cell_specimen_id': cv,
                'ROImask': rv,
                'label': lv,
                'stimulus_name': ik,
                'stimulus_iterations': pv,
                'event_index': ev,
            }]

    # Concatenate data into equal-sized lists
    event_dict = []
    for d in output_data:
        ref_length = d['image'].shape[0]
        assert ref_length == d['label'].shape[0],\
            'Stimuli and neural data do not match.'
        for idx in range(ref_length):
            it_event = {}
            for k, v in d.iteritems():
                if exp_dict['include_targets'][k] == 'split':
                    try:
                        it_event[k] = v[idx]
                    except:
                        raise RuntimeError(
                            'Did you mean to repeat %s per frame?' % k)
                elif exp_dict['include_targets'][k] == 'repeat':
                    it_event[k] = v
                else:
                    raise RuntimeError(
                        'Fucked up packing data into list of dicts.')
            event_dict += [it_event]
    return event_dict, output_rfs, cell_list


def create_example(data_dict, feature_types):
    """Create entry in tfrecords."""
    tf_dict = {}
    for k, v in data_dict.iteritems():
        if k not in feature_types.keys():
            raise RuntimeError('Cannot understand specified feature types.')
        else:
            it_feature_type = feature_types[k]
        if it_feature_type == 'float':
            tf_dict[k] = float_feature(v)
        elif it_feature_type == 'int64':
            tf_dict[k] = int64_feature(v)
        elif it_feature_type == 'string':
            if isinstance(v, basestring):
                # Strings
                tf_dict[k] = bytes_feature(str(v))
            else:
                # Images
                tf_dict[k] = bytes_feature(v.tostring())
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=tf_dict
        )
    )


def prepare_tf_dicts(feature_types, d):
    """Prepare tf data types for loading tf variables."""
    tf_dict = {}
    for k, v in feature_types.iteritems():
        # TODO: Change this interface to more flexibly preallocate shapes.
        if v is 'float' and k in d.keys():
            if isinstance(d[k], list):
                shape = len(d[k])
            elif isinstance(d[k], np.ndarray):
                shape = d[k].shape[0]
            else:
                print 'WARNING: cannot understand the type of feature %s.' % k
                shape = []
        else:
            shape = []
        tf_dict[k] = fixed_len_feature(dtype=v, length=shape)
    return tf_dict


def prepare_data_for_tf_records(
        data_files,
        output_directory,
        rf_dicts,
        cell_order,
        set_name,
        cv_split,
        store_means,
        feature_types,
        cc_repo=None,
        stimuli_key=None,
        ext='tfrecords',
        config=None):
    """Package dict into tfrecords."""
    # TODO: MOVE SLICEING HERE
    if cv_split.keys()[0] == 'random_cv_split':
        cv_inds = np.random.permutation(len(data_files))
        val_len = np.round(
            len(data_files) * cv_split.values()[0]).astype(int)
        val_ind = cv_inds[val_len:]
        train_ind = cv_inds[:val_len]
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
    elif cv_split.keys()[0] == 'cv_split':
        cv_inds = np.arange(len(data_files))
        val_len = np.round(
            len(data_files) * cv_split.values()[0]).astype(int)
        val_ind = cv_inds[val_len:]
        train_ind = cv_inds[:val_len]
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
    elif cv_split.keys()[0] == 'cv_split_single_stim':
        target_stim = cv_split['cv_split_single_stim']['target']
        split = cv_split['cv_split_single_stim']['split']
        unique_stims = np.unique([d['stimulus_name'] for d in data_files])
        if isinstance(target_stim, int):
            # Take the first stimulus
            selection_ind = np.asarray(
                [True if unique_stims[target_stim] in d['stimulus_name']
                    else False for d in data_files])
            trains = np.where(selection_ind)[0]
            train_split = np.floor(len(trains) * split).astype(int)
            train_ind = trains[:train_split]
            val_ind = trains[train_split:]
        else:
            selection_ind = np.asarray(
                [True if target_stim in d['stimulus_name'] else False
                    for d in data_files])
            train_ind = np.asarray(
                [True if target_stim in d['stimulus_name'] else False
                    for d in data_files])
            val_ind = train_ind == False
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
    elif cv_split.keys()[0] == 'split_on_stim':
        train_ind = np.asarray(
            [True if cv_split.values()[0] in d['stimulus_name'] else False
                for d in data_files])
        val_ind = train_ind == False
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
        print 'Split data into Train: %s, Validation: %s.' % (
            np.sum(train_ind),
            np.sum(val_ind))
    elif cv_split.keys()[0] == 'split_on_repeat':
        raise RuntimeError('CV type: split_on_repeat is not yet implemented.')
        # stim_names = [d['stimulus_name'] for d in data_files]
        cv_data = {
            'train': [],
            'val': []
        }

    else:
        raise RuntimeError(
            'Selected crossvalidation %s is not yet implemented.' % cv_split)

    if isinstance(store_means, tuple):
        print 'Converting tuple store_means to a list.'
        store_means = store_means[0]
    means = {k: [] for k in store_means}
    maxs = {k: [] for k in store_means}
    stds = {k: [] for k in store_means}
    for k, v in cv_data.iteritems():
        it_name = os.path.join(
            output_directory,
            '%s_%s.%s' % (set_name, k, ext))
        idx = 0
        assert len(v) > 0, 'Empty validation set found.'
        with tf.python_io.TFRecordWriter(it_name) as tfrecord_writer:
            for idx, d in tqdm(
                    enumerate(v),
                    total=len(v),
                    desc='Encoding %s' % k):
                example = create_example(d, feature_types)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                example = None
                idx += 1
                # Calculate summary stats
            for imk, imv in means.iteritems():
                data_vol = []
                for d in v:
                    data_vol += [np.expand_dims(d[imk], axis=0)]
                data_vol = np.concatenate(data_vol, axis=0)
                if len(data_vol.shape) < 3:
                    means[imk] = np.mean(data_vol)
                    stds[imk] = np.std(data_vol)
                else:
                    means[imk] = np.mean(data_vol, axis=0)
                    stds[imk] = 1.
                maxs[imk] = np.max(data_vol)
                if len(data_vol.shape) < 3:
                    stds[imk] = np.std(data_vol)
                else:
                    stds[imk] = 1.
        mean_file = os.path.join(
            output_directory,
            '%s_%s_means' % (set_name, k))
        num_its = float(len(v))

        # WARNING:
        means = {k: {
            'mean': v / num_its,
            'max': maxs[k],
            'std': stds[k]
            } for k, v in means.iteritems() if not isinstance(v, list)}
        np.savez(mean_file, means)
        print 'Finished encoding: %s' % it_name

    # Save file containing info about the stimuli (i.e. X for X -> Y)
    meta_file = os.path.join(
        output_directory,
        '%s_meta' % (set_name))
    im_size = d[stimuli_key.values()[0]].shape
    tf_load_vars = prepare_tf_dicts(feature_types, d)
    tf_reader = {}
    for ik, iv in v[0].iteritems():
        if isinstance(iv, float) or\
            isinstance(iv, int) or\
                isinstance(iv, basestring):
            it_shape = []
        else:
            it_shape = iv.shape
        # TODO: Align this with numpy typing in experiment declaration.
        tf_reader[ik] = {'dtype': tf.float32, 'reshape': it_shape}
    # tf_reader['image']['reshape'] = cc_repo['model_im_size']
    meta = {
        'im_size': im_size,
        'folds': {k: k for k in cv_data.keys()},
        'tf_dict': tf_load_vars,
        'tf_reader': tf_reader,
        'rf_data': rf_dicts,
        'cell_order': cell_order
    }
    np.save(meta_file, meta)

    # Create a dataloader template file for the cc_bp repo
    if cc_repo is not None:
        dl_file = os.path.join(
            cc_repo['path'],
            '%s.py' % (set_name))
        loader_meta = {
            'NAME': set_name,
            'OUTPUT_SIZE': cc_repo['output_size'],
            'IM_SIZE': im_size,
            'MODEL_IM_SIZE': cc_repo['model_im_size'],
            'META_FILE': meta_file,
            'LOSS_FUNCTION': cc_repo['loss_function'],
            'SCORE_METRIC': cc_repo['score_metric'],
            'PREPROCESS': cc_repo['preprocess'],
        }

        # Create data loader for contextual circuit BP
        create_data_loader_class(
            cc_repo['template_file'], loader_meta, dl_file)

        # Create models for contextual circuit BP
        # summarized_rfs = summarize_rfs(rf_dicts)
        # create_model_files(
        #     output_size=cc_repo['output_size'],
        #     set_name=set_name,
        #     rf_info=summarized_rfs,
        #     template_directory=config.model_template_dir,
        #     model_dir=config.model_struct_dir)


def summarize_rfs(
        rf_dicts,
        x_key='on_center_x',
        y_key='on_center_y',
        k_key='on_width_x'):
    """Return the averages of RF centroids and extents."""
    xs, ys, ks = [], [], []
    for k, v in rf_dicts.iteritems():
        v = v[0]
        if v[x_key] is not None:
            xs += [v[x_key]]
        if v[y_key] is not None:
            ys += [v[y_key]]
        if v[k_key] is not None:
            ks += [v[k_key]]
    h = np.nanmean(ys)
    w = np.nanmean(xs)
    k = np.nanmean(ks)
    assert h != 0, 'No RF height found.'
    assert w != 0, 'No RF width found.'
    assert k != 0, 'No RF kernel found.'
    return {
        'h': h,
        'w': w,
        'k': k

    }


def inclusive_cell_filter(data_dicts, sessions):
    """Filter to only keep cells appearing in all sessions"""
    cell_info = {}
    for d in data_dicts:
        cell_name = d['cell_specimen_id']
        cell_session = d['session']
        if cell_name not in cell_info.keys():
            cell_info[cell_name] = [cell_session]
        else:
            cell_info[cell_name] += [cell_session]
    filtered_data_dicts = []
    for idx, d in enumerate(data_dicts):
        test = [
            k in cell_info[d['cell_specimen_id']]
            for k in np.asarray(sessions)]
        if all(test):
            filtered_data_dicts += [d]
    print 'Filtered %s session exclusive cells (%s/%s remaining).' % (
        len(data_dicts) - len(filtered_data_dicts),
        len(filtered_data_dicts),
        len(data_dicts))
    return filtered_data_dicts


def inclusive_stim_order_filter(data_dicts):
    """Filter data for cells that have inconsistent stimuli lists."""
    # Find unique stimuli
    stim_orders, stim_names = get_stim_names_and_orders(
        data_dicts)

    # Concatenate orders per stimulus
    np_stim_names = np.asarray(stim_names)
    np_stim_orders = np.asarray(stim_orders)
    unique_stimuli = np.unique(stim_names)
    if len(unique_stimuli) == 1 and 'scenes' in unique_stimuli[0]:
        # Handle scenes differently from movies (Find identical orders).
        dm = np.corrcoef(np_stim_orders)
        best_order_counts = np.argmax((np.tril(dm, -1) == 1).sum(1))
        filter_ids = [idx for idx, k in enumerate(
            np_stim_orders) if not np.all(
            k == np_stim_orders[best_order_counts])]
    else:
        filter_list = []
        for stim in unique_stimuli:
            # Loop through stimuli
            stim_idx = np.where(np.asarray(stim) == np_stim_names)[0]
            it_orders = np_stim_orders[stim_idx]
            mat_orders = np.asarray(
                [x.tolist() for x in it_orders]).transpose()
            for order in mat_orders:
                # Loop through order indices
                modal = stats.mode(order)[0][0]
                filter_list += [np.where(order != modal)[0]]
        filter_ids = np.unique(np.concatenate(filter_list))

    # Find unique indices
    filtered_data_dicts = [d for idx, d in enumerate(
        data_dicts) if idx not in filter_ids]
    print 'Filtered %s bad stimulus order cells (%s/%s remaining).' % (
        len(data_dicts) - len(filtered_data_dicts),
        len(filtered_data_dicts),
        len(data_dicts))
    assert len(filtered_data_dicts) > 0,\
        'No data remaining after stimulus order filter.'
    return filtered_data_dicts


def package_dataset(
        config,
        dataset_info,
        output_directory,
        check_stimuli=False):
    """Query and package."""
    dataset_instructions = dataset_info['cross_ref']
    if dataset_instructions == 'rf_coordinate_range':
        # TODO fix this API so it doesn't rely on conditionals.
        data_dicts = data_db.get_cells_all_data_by_rf(
            dataset_info['rf_query'])[0]
    elif dataset_instructions == 'rf_coordinate_range_and_stimuli':
        data_dicts = data_db.get_cells_all_data_by_rf_and_stimuli(
            rfs=dataset_info['rf_query'],
            stimuli=dataset_info['stimuli'],
            sessions=dataset_info['sessions'])[0]
    else:
        # Incorporate more queryies and eventually allow inner-joining on them.
        raise RuntimeError('Other instructions are not yet implemented.')

    if len(data_dicts) == 0:
        raise RuntimeError('Empty cell query.')

    # # Filter cells satisfying only one condition (could be a subquery).
    # data_dicts = inclusive_cell_filter(
    #     data_dicts=data_dicts,
    #     sessions=dataset_info['sessions'])

    # Filter cells that have odd stimulus orderings.
    data_dicts = inclusive_stim_order_filter(data_dicts)

    # Load data
    data_files, rf_dicts, cell_order = load_npzs(
        data_dicts,
        dataset_info,
        stimuli_key=dataset_info['reference_image_key'],
        neural_key=dataset_info['reference_label_key'],
        check_stimuli=check_stimuli
    )

    # Prepare meta file to create a dataset specific data loader
    cc_repo = {
        'template_file': config.cc_template,
        'path': config.cc_data_dir
    }
    for k, v in dataset_info['cc_repo_vars'].iteritems():
        cc_repo[k] = v

    # Add tf_type entries for reference keys
    cat_dict = dict(
        dataset_info['reference_image_key'].items() +
        dataset_info['reference_label_key'].items())
    for k, v in cat_dict.iteritems():
        dataset_info['tf_types'][v] = dataset_info['tf_types'][k]

    # Create tf records and meta files/data loader
    prepare_data_for_tf_records(
        data_files=data_files,
        output_directory=output_directory,
        rf_dicts=rf_dicts,
        cell_order=cell_order,
        set_name=dataset_info['experiment_name'],
        cv_split=dataset_info['cv_split'],
        store_means=dataset_info['store_means'],
        stimuli_key=dataset_info['reference_image_key'],
        feature_types=dataset_info['tf_types'],
        cc_repo=cc_repo,
        config=config)


def main(
        dataset,
        output_directory=None,
        check_stimuli=False):
    """Pull desired experiment cells and encode as tfrecords."""
    assert dataset is not None, 'Name the experiment to process!'
    config = Config()
    if isinstance(dataset, basestring):
        da = dad()[dataset]()
    else:
        da = dataset
    if output_directory is None:
        output_directory = os.path.join(
            config.tf_record_output)
    da['deconv_dir'] = config.deconv_model_dir
    helper_funcs.make_dir(output_directory)
    package_dataset(
        config=config,
        dataset_info=da,
        output_directory=output_directory,
        check_stimuli=check_stimuli)
    # TODO: Incorporate logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        default=None,
        help='Encode an Allen brain institute dataset.')
    parser.add_argument(
        '--output',
        dest='output_directory',
        type=str,
        default=None,
        help='Save tfrecords to this directory.')
    parser.add_argument(
        '--check',
        dest='check_stimuli',
        action='store_true',
        help='Check remaining stimuli before creating dataset.')
    main(**vars(parser.parse_args()))
