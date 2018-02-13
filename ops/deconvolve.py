# import rpy2.robjects.packages
# from c2s import c2s
# from cmt.models import MCGSM
# from c2s import robust_linear_regression
try:
    from deconv_methods import elephant_preprocess, elephant_deconv
except IOerror:
    print 'Unable to import ELEPHANT deconv model.'
try:
    from OASIS.oasis.functions import deconvolve as oasis_deconv
    from OASIS.oasis import oasisAR1, oasisAR2
except IOerror:
    print 'Unable to import OASIS deconv model.'

class deconvolve(object):
    """Wrapper class for deconvolving spikes from Ca2+ data."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Class global variable init."""
        self.data_fps = 30.  # Ca2+ FPS for Allen.
        self.batch_size = 4096
        self.update_params(kwargs)
        self.check_params()
        self.deconvolved_trace = self.deconvolve()

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
        if not hasattr(self, 'neural_trace'):
            raise RuntimeError(
                'You must pass an activity trace.')

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def deconvolve(self, neural_data):
        """Wrapper for deconvolution operations."""
        preproc_op, deconv_op = self.interpret_deconv(self.deconv_method)
        print 'Preprocessing neural data.'
        preproc_data = preproc_op(neural_data)
        print 'Deconvolving neural data.'
        return deconv_op(preproc_data)

    def interpret_deconv(self, method):
        """Wrapper for returning the preprocessing and main operations."""
        if method == 'elephant':
            return (
                elephant_preprocess.preprocess,
                elephant_deconv.deconv)
        elif method == 'lzerospikeinference':
            lzsi = rpy2.robjects.packages.importr("LZeroSpikeInference")
            def lzsi_preprocess(x):
                return x.tolist()
            def lzsi_method(x):
                return lzsi.estimateSpikes(
                    x, **{'gam': 0.998, 'lambda': 8, 'type': "ar1"})
            return (lzsi_preprocess, lzsi_method)
        elif method == 'oasis' or method == 'OASIS':
            def oasis_preprocess(x):
                return double(x)
            def oasis_method(x):
                return oasis_deconv(x)
            return (oasis_preprocess, oasis_deconv)
        elif method == 'c2s':
            def stm_preprocess(x, fps=30.):
                d = []
                for ce in x:
                    d += [{'calcium': ce, 'fps': fps}]
                return c2s.preprocess(d, fps=fps)
            def stm_method(x):
                return c2s.predict(x)
            return (stm_preprocess, stm_method)
        else:
            return None

