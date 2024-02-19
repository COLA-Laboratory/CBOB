import numpy as np
from GPy.core import GP
from GPy import kern
from GPy.likelihoods import HeteroscedasticGaussian



class GPHeteroscedasticRegression(GP):
    """
    Gaussian Process model for heteroscedastic regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf

    NB: This model does not make inference on the noise outside the training set
    """
    def __init__(self, X, Y, kernel=None, mean_function=None, Y_metadata=None):

        Ny = Y.shape[0]

        if Y_metadata is None:
            Y_metadata = {'output_index':np.arange(Ny)[:,None]}
        else:
            assert Y_metadata['output_index'].shape[0] == Ny

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        #Likelihood
        likelihood = HeteroscedasticGaussian(Y_metadata)

        super(GPHeteroscedasticRegression, self).__init__(X, Y, kernel, likelihood, Y_metadata=Y_metadata, mean_function=mean_function)

    def posterior_samples(self, X, size=10, Y_metadata=None, likelihood=None, **predict_kwargs):
        mean, cov = self.predict_noiseless(X, full_cov=True)
        return np.random.multivariate_normal(mean.flatten(), cov, size=size)
