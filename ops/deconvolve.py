"""Deconvolution functions for allen brain observatory data."""
import numpy as np
try:
    from c2s import c2s
    from cmt.models import MCGSM
    from c2s import robust_linear_regression
except:
    print 'Unable to import STM from c2s.'
try:
    import rpy2.robjects.packages
except:
    print 'Unable to import rpy2'
try:
    from deconv_methods import elephant_preprocess, elephant_deconv
except:
    print 'Unable to import ELEPHANT deconv model.'
try:
    from oasis.functions import deconvolve as oasis_deconv
    from oasis import oasisAR1, oasisAR2
except:
    print 'Unable to import OASIS deconv model.'

class deconvolve(object):
    """Wrapper class for deconvolving spikes from Ca2+ data."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, exp_dict):
        """Class global variable init."""
        self.data_fps = 30.  # Ca2+ FPS for Allen.
        self.batch_size = 4096
        self.exp_dict = exp_dict
        self.update_params(exp_dict)
        self.check_params()

    def check_params(self):
        if not hasattr(self, 'deconv_method'):
            print 'Skipping deconvolution'
        if not hasattr(self, 'batch_size'):
            raise RuntimeError(
                'You must pass a batch_size.')
        if not hasattr(self, 'deconv_dir'):
            raise RuntimeError(
                'You must pass a deconv_dir.')
        if not hasattr(self, 'data_fps'):
            raise RuntimeError(
                'You must pass a data_fps.')

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def deconvolve(self, neural_data):
        """Wrapper for deconvolution operations."""
        preproc_op, deconv_op, selection= self.interpret_deconv(self.deconv_method)
        print 'Preprocessing neural data.'
        preproc_data = preproc_op(neural_data)
        print 'Deconvolving neural data.'
        deconv_data = deconv_op(preproc_data)
        if selection is not None:
            deconv_data = deconv_data[selection]
        return deconv_data

    def interpret_deconv(self, method):
        """Wrapper for returning the preprocessing and main operations."""
        if method == 'elephant':
            selection = None
            return (
                elephant_preprocess.preprocess,
                elephant_deconv.deconv,
                selection)
        elif method == 'lzerospikeinference':
            lzsi = rpy2.robjects.packages.importr("LZeroSpikeInference")
            def lzsi_preprocess(x):
                return x.tolist()
            def lzsi_method(x):
                return lzsi.estimateSpikes(
                    x, **{'gam': 0.998, 'lambda': 8, 'type': "ar1"})
            selection = None
            return (lzsi_preprocess, lzsi_method, selection)
        elif method == 'oasis' or method == 'OASIS':
            def oasis_preprocess(x):
                return np.asarray(x).astype(np.float64)
            def oasis_method(x):
                return oasis_deconv(
                    x,
                    g=(None,None),
                    penalty=1)  # ,
                    # optimize_g=5,
                    # max_iter=5)  # denoised, spikes, params
            selection = 1
            return (oasis_preprocess, oasis_method, selection)
        elif method == 'c2s':
            def stm_preprocess(x, fps=30.):
                d = []
                for ce in x:
                    d += [{'calcium': ce, 'fps': fps}]
                return c2s.preprocess(d, fps=fps)
            def stm_method(x):
                return c2s.predict(x)
            selection = None
            return (stm_preprocess, stm_method, selection)
        else:
            return None

