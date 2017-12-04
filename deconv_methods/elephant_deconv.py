"""
Config file: collection of parameters to adjust learning and data pre-processing
"""

from keras.models import Model
from keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, Input
import numpy as np
import os


def norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-7)


def deconv(trace, tracex, params):
    # determines how the time window used as input is positioned around the actual time point
    before_frac, after_frac = 0.25, 0.75

    # width of gaussian kernel used for convolution of spike train
    spike_SD_kernel = 2.0 # half-size of kernel
    spike_size_kernel = 15
    # balanced set of non-spikes / spikes (not used as default)
    balanced_labels = 0
    verbosity = 1

    # key: dataset number, value: list of neuron ids (zero-indexed)
    # if value is None, use all neurons
    datasets_to_train = {
        1: [0,1,2,3,4,5,6,7,8,9,10],
        2: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        3: [0,1,2,3,4,5,6,7,8,9,10,11,12],
        4: [0,1,2,3,4,5],
        5: [0,1,2,3,4,5,6,7,8],
        6: [0,1,2,3,4,5,6,7,8],
        7: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
        8: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        9: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        10: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    }

    spike_threshold = 0.1 # only used if spike_trace contains 'smooth' (only used with 'balanced_labels = 1')
    spike_trace = "spikes_smooth" # "spikes"
    calcium_trace = "calcium" # "calcium" # "calcium_smooth", "calcium_smooth_norm"

    loss_function = 'mean_squared_error'
    optimizer = 'Adagrad'
    batch_size = 256
    nr_of_epochs = 15

    datasets_to_test = {
        1: [0,1,2,3,4,5,6,7,8,9,10],
        2: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        3: [0,1,2,3,4,5,6,7,8,9,10,11,12],
        4: [0,1,2,3,4,5],
        5: [0,1,2,3,4,5,6,7,8],
        6: [0,1,2,3,4,5,6,7,8],
        7: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
        8: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        9: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        10: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    }


    datasets = np.array([  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,
             2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,
             2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   3.,
             3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,   3.,
             3.,   4.,   4.,   4.,   4.,   4.,   4.,   5.,   5.,   5.,   5.,
             5.,   5.,   5.,   5.,   5.,   6.,   6.,   6.,   6.,   6.,   6.,
             6.,   6.,   6.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,
             7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,
             7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,
             7.,   7.,   7.,   7.,   7.,   7.,   7.,   8.,   8.,   8.,   8.,
             8.,   8.,   8.,   8.,   8.,   8.,   8.,   8.,   8.,   8.,   8.,
             8.,   8.,   8.,   8.,   8.,   8.,   9.,   9.,   9.,   9.,   9.,
             9.,   9.,   9.,   9.,   9.,   9.,   9.,   9.,   9.,   9.,   9.,
             9.,   9.,   9.,   9.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,
            10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,
            10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.])
                                    
    neurons = np.array(
        [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
             0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
            11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,   0.,
             1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
            12.,   0.,   1.,   2.,   3.,   4.,   5.,   0.,   1.,   2.,   3.,
             4.,   5.,   6.,   7.,   8.,   0.,   1.,   2.,   3.,   4.,   5.,
             6.,   7.,   8.,   0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,
             8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,
            19.,  20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,
            30.,  31.,  32.,  33.,  34.,  35.,  36.,   0.,   1.,   2.,   3.,
             4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,
            15.,  16.,  17.,  18.,  19.,  20.,   0.,   1.,   2.,   3.,   4.,
             5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.,
            16.,  17.,  18.,  19.,   0.,   1.,   2.,   3.,   4.,   5.,   6.,
             7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,
            18.,  19.,  20.,  21.,  22.,  23.,  24.,  25.,  26.])

    """"

    Define the model using the function API of Keras.

    """

    windowsize = 128
    conv_filter = Conv1D
    filter_size = (41, 21, 7)
    filter_number = (50, 60, 70)
    dense_expansion = 300

    if verbosity:
        print("setup training data with {} datasets".format(
            len(datasets_to_train)))

    inputs = Input(shape=(windowsize, 1))

    outX = conv_filter(
        filter_number[0], filter_size[0], strides=1, activation='relu')(inputs)
    outX = conv_filter(
        filter_number[1], filter_size[1], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)
    outX = conv_filter(
        filter_number[2], filter_size[2], activation='relu')(outX)
    outX = MaxPooling1D(2)(outX)

    outX = Dense(dense_expansion, activation='relu')(outX)
    outX = Flatten()(outX)
    predictions = Dense(1, activation='linear')(outX)
    model = Model(inputs=[inputs], outputs=predictions)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.load_weights(
        os.path.join(
            params.deconv_dir,
            'elephant',
            'model1.h5'))

    Ypredict = np.zeros((tracex.shape[0]-windowsize, tracex.shape[1]))
    for k in range(0, trace.shape[1]):
        if np.mod(k, 20) == 0:
            print(
                'Predicting spikes for neuron %s out of %s' % (
                    k+1,
                    trace.shape[1]))
        x1x = tracex[:, k]
        idx = ~np.isnan(x1x)
        calcium_traceX = norm(x1x[idx])
        # initialize the prediction vector
        XX = np.zeros((
            calcium_traceX.shape[0]-windowsize, windowsize, 1),
            dtype=np.float32)
        for jj in range(0, (calcium_traceX.shape[0]-windowsize)):
            XX[jj, :, 0] = calcium_traceX[jj:(jj+windowsize)]
        A = model.predict(XX, params['batch_size'])
        Ypredict[idx[0:len(idx)-windowsize], k] = A[:, 0]
    return Ypredict
