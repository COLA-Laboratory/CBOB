from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
import numpy as np
from scipy.stats import norm
from utils import SobolDesign


class AcquisitionEICB_NOCLIP(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, BOmodel, space, optimizer, cost_withGradients=None, jitter=0.01, threshold=0., balance=None,
                 beta=2., sampling_num=1000, classification=False):
        self.optimizer = optimizer
        super(AcquisitionEICB_NOCLIP, self).__init__(BOmodel, space, optimizer)
        self.cons_num = BOmodel.constraint_num
        self.BOModel = BOmodel
        self.jitter = jitter
        self.threshold = threshold
        self.beta = beta
        self.sampling_num = sampling_num
        if balance is None:
            self.balance = np.zeros((self.cons_num,))
        else:
            self.balance = balance
        self.classification = classification
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('EIC acquisition does now make sense with cost at present. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, x):

        self.model_cons = self.BOModel.model_cons
        m, s = self.BOModel.model_obj.predict(x)
        fmin = self.BOModel.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu_ei = s * (u * Phi + phi)

        # the probability of feasibility acquisition
        prob = 1.0
        for i in range(self.cons_num):
            mean, var = self.model_cons[i].predict_noiseless(x)
            std = np.sqrt(var)

            cdf = norm.cdf(self.threshold, loc=mean, scale=std)
            cdf_lower = norm.cdf(self.threshold - self.beta * np.sqrt(var), loc=mean, scale=std)
            cdf_upper = norm.cdf(self.threshold + self.beta * np.sqrt(var), loc=mean, scale=std)

            prob = prob * (cdf + cdf * (cdf_upper - cdf_lower))


        # return the product of EI and PROB
        return f_acqu_ei * prob

    def _compute_acq_withGradients(self, x):
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        raise NotImplementedError()


    def compute_EI(self, x):
        self.model_cons = self.BOModel.model_cons
        m, s = self.BOModel.model_obj.predict(x)
        fmin = self.BOModel.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu_ei = s * (u * Phi + phi)

        return f_acqu_ei

    def compute_POF(self, x):
        self.model_cons = self.BOModel.model_cons

        # the probability of feasibility acquisition
        prob = 1.0
        for i in range(self.cons_num):
            mean, var = self.model_cons[i].predict_noiseless(x)
            std = np.sqrt(var)

            cdf = norm.cdf(self.threshold, loc=mean, scale=std)
            cdf_lower = norm.cdf(self.threshold - self.beta * np.sqrt(var), loc=mean, scale=std)
            cdf_upper = norm.cdf(self.threshold + self.beta * np.sqrt(var), loc=mean, scale=std)

            prob = prob * (cdf + cdf * (cdf_upper - cdf_lower))
        return prob

    def sampling_next_point(self):
        seed = int(np.random.uniform(1, 1e5, 1))
        sampler = SobolDesign(self.space)
        sample_points = sampler.get_samples(10000, seed)
        acq = self._compute_acq(sample_points)
        best_acq = np.max(acq, axis=0)
        suggested_point = np.argmax(acq, axis=0)
        return sample_points[suggested_point], best_acq