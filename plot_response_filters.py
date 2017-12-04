import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from allen_config import Allen_Brain_Observatory_Config
from matplotlib import pyplot as plt
from matplotlib import gridspec
# import pandas as pd
# from ggplot import *
import seaborn as sns
from allensdk.brain_observatory.receptive_field_analysis import receptive_field


def save_mosaic(
        maps,
        output,
        title='Mosaic',
        rc=None,
        cc=None):
    if rc is None:
        rc = np.ceil(np.sqrt(len(maps))).astype(int)
        cc = np.ceil(np.sqrt(len(maps))).astype(int)
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=20)
    gs1 = gridspec.GridSpec(rc, cc)
    gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
    for idx, im in enumerate(maps):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(im.squeeze())
    plt.savefig(output)
    plt.savefig(output.split('.')[0] + '.pdf')


def plot_fits(
        experiment='760_cells_2017_11_04_16_29_09',
        query_db=False,
        num_models=3,
        template_exp='ALLEN_selected_cells_1',
        process_pnodes=False):
    """Plot fits across the RF.
    experiment: Name of Allen experiment you're plotting.
    query_db: Use data from DB versus data in Numpys.
    num_models: The number of architectures you're testing.
    template_exp: The name of the contextual_circuit model template used."""

    if process_pnodes:
        from pnodes_declare_datasets_loop import query_hp_hist, sel_exp_query
    else:
        from declare_datasets_loop import query_hp_hist, sel_exp_query

    main_config = Allen_Brain_Observatory_Config()
    sys.path.append(main_config.cc_path)
    sys.path.append(os.path.join(main_config.cc_path, 'ops'))
    from db import credentials
    import data_loader
    db_config = credentials.postgresql_connection()
    files = glob(
        os.path.join(
            main_config.multi_exps,
            experiment, '*.npz'))
    out_data, xs, ys, preds = [], [], [], []
    perfs, model_types, exps, arg_perf = [], [], [], []
    count = 0
    for f in files:
        data = np.load(f)
        d = {
            'x': data['rf_data'].item()['on_center_x'],
            'y': data['rf_data'].item()['on_center_y'],
            # x: files['dataset_method'].item()['x_min'],
            # y: files['dataset_method'].item()['y_min'],
        }
        exp_name = {
            'experiment_name': data['dataset_method'].item()[
                'experiment_name']}
        if query_db:
            perf = query_hp_hist(
                exp_name['experiment_name'],
                db_config=db_config)
            if perf is None:
                print 'No fits for: %s' % exp_name['experiment_name']
            else:
                raise NotImplementedError
                d['perf'] = perf
                d['max_val'] = np.max(perf)
                out_data += [d]
                xs += [np.round(d['x'])]
                ys += [np.round(d['y'])]
                perfs += [np.max(d['perf'])]
                count += 1
        else:
            data_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_losses.npy'))  # Scores has preds, labels has GT
            pred_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_preds.npy'))  # Scores has preds, labels has GT
            for gd, pd in zip(data_files, pred_files):
                mt = gd.split(
                    os.path.sep)[-1].split(
                        template_exp + '_')[-1].split('_' + 'val')[0]
                it_data = np.load(gd).item()
                pd_data = np.load(pd).item()
                import ipdb;ipdb.set_trace()
                sinds = np.asarray(it_data.keys())[np.argsort(it_data.keys())]
                sit_data = [it_data[idx] for idx in sinds]
                d['perf'] = sit_data
                d['max_val'] = np.max(sit_data)
                d['max_idx'] = np.argmax(sit_data)
                d['mt'] = mt
                out_data += [d]
                xs += [np.round(d['x'])]
                ys += [np.round(d['y'])]
                perfs += [np.max(sit_data)]
                arg_perf += [np.argmax(sit_data)]
                exps += [gd.split(os.path.sep)[-2]]
                model_types += [mt]
                count += 1

    # Package as a df
    xs = np.round(np.asarray(xs)).astype(int)
    ys = np.round(np.asarray(ys)).astype(int)
    perfs = np.asarray(perfs)
    arg_perf = np.asarray(arg_perf)
    exps = np.asarray(exps)
    model_types = np.asarray(model_types)

    # Filter to only keep top-scoring values at each x/y (dirty trick)
    fxs, fys, fperfs, fmodel_types, fexps, fargs = [], [], [], [], [], []
    xys = np.vstack((xs, ys)).transpose()
    cxy = np.ascontiguousarray(  # Unique rows
        xys).view(
        np.dtype((np.void, xys.dtype.itemsize * xys.shape[1])))
    _, idx = np.unique(cxy, return_index=True)
    uxys = xys[idx]
    for xy in uxys:
        sel_idx = (xys == xy).sum(axis=-1) == 2
        sperfs = perfs[sel_idx]
        sexps = exps[sel_idx]
        sargs = arg_perf[sel_idx]
        sel_mts = model_types[sel_idx]
        bp = np.argmax(sperfs)
        fxs += [xy[0]]
        fys += [xy[1]]
        fperfs += [sperfs[bp]]
        fargs += [sargs[bp]]
        fmodel_types += [sel_mts[bp]]
        fexps += [sexps[bp]]
    xs = np.asarray(fxs)
    ys = np.asarray(fys)
    perfs = np.asarray(fperfs)
    arg_perf = np.asarray(fargs)
    exps = np.asarray(fexps)
    model_types = np.asarray(fmodel_types)
    umt, model_types_inds = np.unique(model_types, return_inverse=True)

    # Get weights for the top-n fitting models of each type
    top_n = 1
    target_layer = 'conv2d'
    it_perfs = perfs[model_types == target_layer]
    it_exps = exps[model_types == target_layer]
    # it_args = arg_perf[model_types == target_layer]
    sorted_perfs = np.argsort(it_perfs)[::-1][:top_n]
    for idx in sorted_perfs:
        perf = sel_exp_query(
            experiment_name=it_exps[idx],
            model=target_layer,
            db_config=db_config)
        # perf_steps = np.argsort([v['training_step'] for v in perf])[::-1]
        perf_steps = [v['validation_loss'] for v in perf]
        max_score = np.max(perf_steps)
        arg_perf_steps = np.argmax(perf_steps)
        sel_model = perf[arg_perf_steps]  # perf_steps[it_args[idx]]]
        print 'Using %s' % sel_model
        model_file = sel_model['ckpt_file'].split('.')[0]
        model_ckpt = '%s.ckpt-%s' % (
            model_file,
            model_file.split(os.path.sep)[-1].split('_')[-1])
        model_meta = '%s.meta' % model_ckpt

        # Pull stimuli
        stim_dir = os.path.join(
            main_config.tf_record_output, 
            sel_model['experiment_name'])
        stim_files = glob(stim_dir + '*')
        stim_meta_file = [x for x in stim_files if 'meta' in x][0]
        stim_val_data = [x for x in stim_files if 'val.tfrecords' in x][0]
        stim_val_mean = [x for x in stim_files if 'val_means' in x][0]
        assert stim_meta_file is not None
        assert stim_val_data is not None
        assert stim_val_mean is not None
        stim_meta_data = np.load(stim_meta_file).item()
        stim_mean_data = np.load(
            stim_val_mean).items()[0][1].item()['image']['mean']
        # Store sparse noise for reference
        sparse_rf_on = {
            'center_x': stim_meta_data.get('on_center_x', None),
            'center_y': stim_meta_data.get('on_center_y', None),
            'width_x': stim_meta_data.get('on_width_x', None),
            'width_y': stim_meta_data.get('on_width_y', None),
            'distance': stim_meta_data.get('on_distance', None),
            'area': stim_meta_data.get('on_area', None),
            'rotation': stim_meta_data.get('on_rotation', None),
        }
        sparse_rf_off = {
            'center_x': stim_meta_data.get('off_center_x', None),
            'center_y': stim_meta_data.get('off_center_y', None),
            'width_x': stim_meta_data.get('off_width_x', None),
            'width_y': stim_meta_data.get('off_width_y', None),
            'distance': stim_meta_data.get('off_distance', None),
            'area': stim_meta_data.get('off_area', None),
            'rotation': stim_meta_data.get('off_rotation', None),
        }   
        sparse_rf = {'on': sparse_rf_on, 'off': sparse_rf_off}
        import ipdb;ipdb.set_trace()

        save_mosaic(
            maps=filts[0].squeeze().transpose(2, 0, 1),
            output='%s_filters' % target_layer,
            rc=8,
            cc=4,
            title='%s filters for cell where rho=%s' % (
                target_layer,
                np.around(max_score, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='760_cells_2017_11_05_16_36_55',
        dest='experiment',
        help='Name of experiment.')
    parser.add_argument(
        '--pnode',
        action='store_true',
        dest='process_pnodes',
        help='Access the pnode DB.')
    args = parser.parse_args()
    plot_fits(**vars(args))
