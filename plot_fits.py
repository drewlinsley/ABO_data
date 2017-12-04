import os
import sys
import argparse
import numpy as np
from glob import glob
from allen_config import Allen_Brain_Observatory_Config
# from declare_datasets_loop import query_hp_hist
from matplotlib import pyplot as plt
import pandas as pd
# from ggplot import *
import seaborn as sns
import joypy as jp
from tqdm import tqdm


def create_joyplot(all_perfs, all_model_types, output_name):
    """Creates jd hists."""

    df = pd.DataFrame(
        np.vstack((all_perfs, all_model_types)).transpose(),
        columns=['x', 'g']) 

    # Initialize the FacetGrid object
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    pal = sns.hls_palette(3, l=0.7, s=0.2)  # , l=.3, s=.8)
    # pal = sns.hls_palette(3, l=0.3, s=0.8)  # l=0.7, s=0.2)  # , l=.3, s=.8)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=2, size=2, palette=pal)
    plt.xlim([-0.2, 1])

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.0001)
    g# .map(sns.kdeplot, "x", clip_on=False, color="w", lw=1.5, bw=.0001,)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    # def label(x, color, label):
    #     ax = plt.gca()
    #    ax.text(-0.1, .2, label, fontweight="bold", color=color, 
    #          ha="left", va="center", transform=ax.transAxes)

    # g.map(label, "x")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play will with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.savefig(output_name)
    plt.show()


def plot_fits(
        experiment='760_cells_2017_11_04_16_29_09',
        query_db=False,
        num_models=3,
        template_exp='ALLEN_selected_cells_1'):
    """Plot fits across the RF.


    experiment: Name of Allen experiment you're plotting.
    query_db: Use data from DB versus data in Numpys.
    num_models: The number of architectures you're testing.
    template_exp: The name of the contextual_circuit model template used."""
    main_config = Allen_Brain_Observatory_Config()
    sys.path.append(main_config.cc_path)
    from db import credentials
    db_config = credentials.postgresql_connection()
    files = glob(
        os.path.join(
            main_config.multi_exps,
            experiment, '*.npz'))
    out_data, xs, ys, perfs, model_types = [], [], [], [], []
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
            perf, mt = query_hp_hist(exp_name, pass_creds=db_config)
            if perf is None:
                print 'No fits for: %s' % exp_name['experiment_name']
            else:
                d['perf'] = perf
                d['max_val'] = np.max(perf)
                d['mt'] = mt
                out_data += [d]
                xs += [np.round(d['x'])]
                ys += [np.round(d['y'])]
                perfs += [np.max(d['perf'])]
                model_types += [mt]
                count += 1
        else:
            data_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_losses.npy'))  # Scores has preds, labels has GT
            score_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_scores.npy'))  # Scores has preds, labels has GT
            lab_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_labels.npy'))  # Scores has preds, labels has GT
            for gd, sd, ld in tqdm(zip(data_files, score_files, lab_files), total=len(data_files)):
                mt = gd.split(
                    os.path.sep)[-1].split(
                        template_exp + '_')[-1].split('_' + 'val')[0]
                it_data = np.load(gd).item()
                lds = np.load(ld).item()
                sds = np.load(sd).item()
                # it_data = {k: np.corrcoef(lds[k], sds[k])[0, 1] for k in sds.keys()}
                sinds = np.asarray(it_data.keys())[np.argsort(it_data.keys())]
                sit_data = [it_data[idx] for idx in sinds]
                d['perf'] = sit_data
                d['max_val'] = np.max(sit_data)
                d['mt'] = mt
                out_data += [d]
                xs += [np.round(d['x'])]
                ys += [np.round(d['y'])]
                perfs += [np.max(sit_data)]
                model_types += [mt]
                count += 1

    # Package as a df
    xs = np.round(np.asarray(xs)).astype(int)
    ys = np.round(np.asarray(ys)).astype(int)
    all_perfs = np.asarray(perfs)
    all_model_types = np.asarray(model_types)

    # Medians per layer
    print 'Dog med: %s | Dog max:%s' % (
        np.median(all_perfs[all_model_types == ['DoG']]),
        np.max(all_perfs[all_model_types == ['DoG']]))
    print 'Conv med: %s | Conv max:%s' % (
        np.median(all_perfs[all_model_types == ['conv2d']]),
        np.max(all_perfs[all_model_types == ['conv2d']]))
    print 'Sep med: %s | Sep max:%s' % (
        np.median(all_perfs[all_model_types == ['sep_conv2d']]),
        np.max(all_perfs[all_model_types == ['sep_conv2d']]))


    # Create a joyplot
    create_joyplot(
        all_perfs=all_perfs,
        all_model_types=all_model_types,
        output_name='joy_%s.pdf' % experiment)

    # Filter to only keep top-scoring values at each x/y (dirty trick)
    fxs, fys, fperfs, fmodel_types = [], [], [], []
    xys = np.vstack((xs, ys)).transpose()
    cxy = np.ascontiguousarray(xys).view(np.dtype((np.void, xys.dtype.itemsize * xys.shape[1])))
    _, idx = np.unique(cxy, return_index=True)
    uxys= xys[idx]
    for xy in uxys:
        sel_idx = (xys == xy).sum(axis=-1) == 2
        sperfs = all_perfs[sel_idx]
        sel_mts = all_model_types[sel_idx]
        bp = np.argmax(sperfs)
        fxs += [xy[0]]
        fys += [xy[1]]
        fperfs += [sperfs[bp]]
        fmodel_types += [sel_mts[bp]]
    xs = np.asarray(fxs)
    ys = np.asarray(fys)
    perfs = np.asarray(fperfs)
    model_types = np.asarray(fmodel_types) 
    umt, model_types_inds = np.unique(model_types, return_inverse=True)
    cps = [
        'Reds',
        'Greens',
        'Blues'
        # sns.cubehelix_palette(n_colors=100, as_cmap=True, start=10.),
        # sns.cubehelix_palette(n_colors=100, as_cmap=True, rot=-.4),
        # sns.cubehelix_palette(n_colors=100, as_cmap=True, rot=.1, start=2.8)
        ]
    # colored_mts = np.asarray(cps)[model_types_inds].tolist()

    # cps = ["Blues", "Reds", "Greens"]
    f, ax = plt.subplots(figsize=(15,7))
    p = []
    for idx, (imt, cp) in enumerate(zip(umt, cps)):
        it_xs = xs[model_types == imt]
        it_ys = ys[model_types == imt]
        it_perfs = perfs[model_types == imt]
        # df = pd.DataFrame(np.vstack((it_xs, it_ys, it_perfs)).transpose(), columns=['x', 'y', 'perf'])
        # p += [sns.lmplot(x='x', y='y', hue='perf', palette=cp, data=df, markers='s', fit_reg=False, x_jitter=1, y_jitter=1)]
        p += [ax.scatter(it_xs, it_ys, c=it_perfs, cmap=cp, label=imt, vmin=0, vmax=1)]

    # TODO: Make sure we are truly plotting the winning model at each spot... I don't think this is happeneing
    # for x, y, p, co, mt in zip(xs, ys, perfs, colored_mts, model_types):
    #     ax.scatter(x, y, c=p, cmap=co, label=mt)
    #     # p += [ax.scatter(x, y, c=p, cmap=co, label=mt)]
    plt.legend()
    legend = ax.get_legend()
    legend.legendHandles[0].set_color(plt.cm.Reds(.8))
    legend.legendHandles[1].set_color(plt.cm.Greens(.8))
    legend.legendHandles[2].set_color(plt.cm.Blues(.8))
    plt.colorbar(p[-1], cmap='Greys')
    # [plt.colorbar(
    #    ip, ticks=np.linspace(
    #         -.1, 1, 100, endpoint=True)) for ip in p]
    # plt.xlim([-100, 1500 * 10])
    # plt.ylim([-100, 1000 * 10])
    plt.xlabel('X-axis degrees of visual-angle')
    plt.ylabel('Y-axis degrees of visual-angle')
    plt.title(
        '%s/%s models finished for %s cells.\nEach point is the winning model\'s validation fit at a neuron\'s derived RF.' % (
            count, len(files) * num_models, len(files)))
    plt.savefig('%s fit_scatters.pdf' % experiment)
    plt.show()
    plt.close(f)

    cps = [
        'Red',
        'Green',
        'Blue']
    f, axs = plt.subplots(1, 3, figsize=(15,7))
    for imt, cp, ax in zip(umt, cps, axs):
        it_perfs = perfs[model_types == imt]
        sns.distplot(it_perfs, bins=100, rug=False, kde=True, ax=ax, color=cp, label=imt, hist_kws={"range": [-0.1, 1]})
        ax.set_title(imt)
    plt.xlabel('Pearson correlation with validation data')
    plt.ylabel('Frequency')
    plt.savefig('fit_hists.png')
    plt.show()
    plt.close(f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='760_cells_2017_11_05_16_36_55',
        dest='experiment',
        help='Name of experiment.')
    args = parser.parse_args()
    plot_fits(**vars(args))
