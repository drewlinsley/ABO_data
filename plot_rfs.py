"""Script for plotting cells and figuring out which to pull for experiments."""

import cv2
from matplotlib import pyplot as plt
from allen_config import Allen_Brain_Observatory_Config
import numpy as np
from data_db import data_db
import argparse
import matplotlib.patches as mpatches
from tqdm import tqdm


def queries_list():
    """Dictionary with all cell queries to run."""
    queries = [
        # {  # Get a range
        #   'x_min': 9,
        #   'x_max': 50,
        #   'y_min': 9,
        #   'y_max': 20,
        # },
        # {
        #     'x_min': 50,
        #     'x_max': 40,
        #     'y_min': 80,
        #     'y_max': 90,
        # },
        # {
        #     'x_min': 20,
        #     'x_max': 30,
        #     'y_min': 40,
        #     'y_max': 50,
        # },
        # {  # Get all
        #     'x_min': 40,
        #     'x_max': 70,
        #     'y_min': 20,
        #     'y_max': 50,
        # },
        [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
    ]
    query_labels = [
        # 'center x in (9,50), y in (9,20)',
        'Selected cell density.',
        'Elipse-extended cell density.',
        'All cells with on RF'
    ]
    return queries, query_labels


def main(
        filter_by_stim=[
            'natural_movie_one',
            'natural_movie_two',
            # 'natural_movie_three'
        ],
        sessions=[
            'three_session_C',
            'three_session_C2'
        ],
        plot_heatmap=False,
        color='b',
        kernel=(5, 5)):
    """Main script for plotting cells by RFs."""
    main_config = Allen_Brain_Observatory_Config()
    queries, query_labels = queries_list()
    if filter_by_stim is not None:
        print 'Pulling cells by their RFs and stimulus: %s.' % filter_by_stim
        all_data_dicts = []
        for q in queries:
            all_data_dicts += [data_db.get_cells_all_data_by_rf_and_stimuli(
                rfs=q,
                stimuli=filter_by_stim,
                sessions=sessions)]
    else:
        print 'Pulling cells by their RFs.'
        all_data_dicts = data_db.get_cells_all_data_by_rf(queries)

    # Make sure there are no null RFs
    visual_space_h = np.floor(main_config.LSN_size_in_deg['height'])
    visual_space_w = np.floor(main_config.LSN_size_in_deg['width'])
    if plot_heatmap:
        for data_dicts, label in tqdm(
                zip(
                    all_data_dicts,
                    query_labels),
                total=len(all_data_dicts),
                desc='Plotting cell heatmap'):

            canvas = np.zeros((int(visual_space_h), int(visual_space_w)))
            for dat in data_dicts[0]:
                try:
                    canvas[int(dat['on_center_y']), int(dat['on_center_x'])] += 1
                except:
                    import ipdb;ipdb.set_trace()
            canvas = cv2.GaussianBlur(canvas, kernel, 0)
            f = plt.figure()
            plt.title('%s %s cells' % (label, len(data_dicts[0])))
            plt.imshow(canvas)
            plt.show()
            plt.close(f)

    for data_dicts, label in tqdm(
            zip(
                all_data_dicts,
                query_labels),
            total=len(all_data_dicts),
            desc='Plotting cell centroids'):
        fig = plt.figure()
        for dat in data_dicts[0]:
            plt.scatter(dat['on_center_y'], dat['on_center_x'])
        plt.xlim(0, visual_space_w)
        plt.ylim(0, visual_space_h)
        plt.title('%s %s cells' % (label, len(data_dicts[0])))
        plt.show()

    for data_dicts, label in tqdm(
            zip(
                all_data_dicts,
                query_labels),
            total=len(all_data_dicts),
            desc='Plotting cell elipsoids'):
        fig, ax = plt.subplots()
        ax.set_xlim(0, visual_space_w)
        ax.set_ylim(0, visual_space_h)
        for dat in data_dicts[0]:
            xy = (dat['on_center_y'], dat['on_center_x'])
            width = 3 * np.abs(dat['on_width_y'])
            height = 3 * np.abs(dat['on_width_x'])
            angle = dat['on_rotation']
            if np.logical_not(any(np.isnan(xy))):
                ellipse = mpatches.Ellipse(
                    xy,
                    width=width,
                    height=height,
                    angle=angle,
                    lw=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.1)
                ax.add_artist(ellipse)
        plt.xlim(0, visual_space_w)
        plt.ylim(0, visual_space_h)
        plt.title(label)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot_heatmap',
        action='store_true',
        dest='plot_heatmap',
        help='Plot a blurred, heatmap version of the centroids.')
    args = parser.parse_args()
    main(**vars(args))
