import numpy as np
from detect_events import detect_events
from utilities import get_A, get_A_blur, get_shuffle_matrix, get_components, dict_generator
from statsmodels.sandbox.stats.multicomp import multipletests


def events_to_pvalues_no_fdr_correction(data, event_vector, A, number_of_shuffles=5000, response_detection_error_std_dev=.1, seed=1):

    number_of_pixels = A.shape[0] / 2

    # Initializations:
    number_of_events = event_vector.sum()
    np.random.seed(seed)

    shuffle_data = get_shuffle_matrix(data, event_vector, A, number_of_shuffles=number_of_shuffles, response_detection_error_std_dev=response_detection_error_std_dev)

    # Build list of p-values:
    response_triggered_stimulus_vector = A.dot(event_vector)/number_of_events
    p_value_list = []
    for pi in range(2*number_of_pixels):
        curr_p_value = 1-(shuffle_data[pi, :] < response_triggered_stimulus_vector[pi]).sum()*1./number_of_shuffles
        p_value_list.append(curr_p_value)

    return np.array(p_value_list)


def compute_receptive_field(data, cell_index, stimulus, **kwargs):

    alpha = kwargs.pop('alpha')

    event_vector = detect_events(data, cell_index, stimulus)

    A_blur = get_A_blur(data, stimulus)
    number_of_pixels = A_blur.shape[0]/2

    pvalues = events_to_pvalues_no_fdr_correction(data, event_vector, A_blur, **kwargs)


    stimulus_table = data.get_stimulus_table(stimulus)
    stimulus_template = data.get_stimulus_template(stimulus)[stimulus_table['frame'].values, :, :]
    s1, s2 = stimulus_template.shape[1], stimulus_template.shape[2]
    pvalues_on, pvalues_off = pvalues[:number_of_pixels].reshape(s1, s2), pvalues[number_of_pixels:].reshape(s1, s2)



    fdr_corrected_pvalues = multipletests(pvalues, alpha=alpha)[1]

    fdr_corrected_pvalues_on = fdr_corrected_pvalues[:number_of_pixels].reshape(s1, s2)
    _fdr_mask_on = np.zeros_like(pvalues_on, dtype=np.bool)
    _fdr_mask_on[fdr_corrected_pvalues_on < alpha] = True
    components_on, number_of_components_on = get_components(_fdr_mask_on)

    fdr_corrected_pvalues_off = fdr_corrected_pvalues[number_of_pixels:].reshape(s1, s2)
    _fdr_mask_off = np.zeros_like(pvalues_off, dtype=np.bool)
    _fdr_mask_off[fdr_corrected_pvalues_off < alpha] = True
    components_off, number_of_components_off = get_components(_fdr_mask_off)

    A = get_A(data, stimulus)
    A_blur = get_A_blur(data, stimulus)

    response_triggered_stimulus_field = A.dot(event_vector)
    response_triggered_stimulus_field_on = response_triggered_stimulus_field[:number_of_pixels].reshape(s1, s2)
    response_triggered_stimulus_field_off = response_triggered_stimulus_field[number_of_pixels:].reshape(s1, s2)

    response_triggered_stimulus_field_convolution = A_blur.dot(event_vector)
    response_triggered_stimulus_field_convolution_on = response_triggered_stimulus_field_convolution[:number_of_pixels].reshape(s1, s2)
    response_triggered_stimulus_field_convolution_off = response_triggered_stimulus_field_convolution[number_of_pixels:].reshape(s1, s2)

    on_dict = {'pvalues':{'data':pvalues_on},
               'fdr_corrected':{'data':fdr_corrected_pvalues_on, 'attrs':{'alpha':alpha, 'min_p':fdr_corrected_pvalues_on.min()}},
               'fdr_mask': {'data':components_on, 'attrs':{'alpha':alpha, 'number_of_components':number_of_components_on, 'number_of_pixels':components_on.sum(axis=1).sum(axis=1)}},
               'rts_convolution':{'data':response_triggered_stimulus_field_convolution_on},
               'rts': {'data': response_triggered_stimulus_field_on}
               }
    off_dict = {'pvalues':{'data':pvalues_off},
               'fdr_corrected':{'data':fdr_corrected_pvalues_off, 'attrs':{'alpha':alpha, 'min_p':fdr_corrected_pvalues_off.min()}},
               'fdr_mask': {'data':components_off, 'attrs':{'alpha':alpha, 'number_of_components':number_of_components_off, 'number_of_pixels':components_off.sum(axis=1).sum(axis=1)}},
               'rts_convolution': {'data': response_triggered_stimulus_field_convolution_off},
               'rts': {'data': response_triggered_stimulus_field_off}
                }

    result_dict = {'event_vector': {'data':event_vector, 'attrs':{'number_of_events':event_vector.sum()}},
                   'on':on_dict,
                   'off':off_dict,
                   'attrs':{'cell_index':cell_index, 'stimulus':stimulus}}

    return result_dict
