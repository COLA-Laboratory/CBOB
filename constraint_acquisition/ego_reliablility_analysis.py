from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.util.general import get_quantiles
import numpy as np
from scipy.stats import norm

class AcquisitionEGOReliabilityAnalysis(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, BOmodel, space, optimizer, cost_withGradients=None, jitter=0.01, threshold=0., balance=.2, beta=2.):
        self.optimizer = optimizer
        super(AcquisitionEGOReliabilityAnalysis, self).__init__(BOmodel, space, optimizer)
        self.cons_num = BOmodel.constraint_num
        self.BOModel = BOmodel
        self.jitter = jitter
        self.threshold = threshold
        self.balance = balance
        self.beta = beta
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('EIC acquisition does now make sense with cost at present. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, x):

        self.model_cons = self.BOModel.model_cons

        # the probability of feasibility acquisition
        prob = 1.0
        for i in range(self.cons_num):
            mean, var = self.model_cons[i].predict_noiseless(x)
            std = np.sqrt(var)

            cdf = norm.cdf(self.threshold, loc=mean, scale=std)
            cdf_lower = norm.cdf(self.threshold - self.beta * np.sqrt(var), loc=mean, scale=std)
            cdf_upper = norm.cdf(self.threshold + self.beta * np.sqrt(var), loc=mean, scale=std)

            pdf = norm.pdf(self.threshold, loc=mean, scale=std)
            pdf_lower = norm.pdf(self.threshold - self.beta * np.sqrt(var), loc=mean, scale=std)
            pdf_upper = norm.pdf(self.threshold + self.beta * np.sqrt(var), loc=mean, scale=std)

            cdf = np.clip(cdf, 1e-16, 1.)
            cdf_lower = np.clip(cdf_lower, 1e-16, 1.)
            cdf_upper = np.clip(cdf_upper, 1e-16, 1.)

            pdf = np.clip(pdf, 1e-16, 1e10)
            pdf_lower = np.clip(pdf_lower, 1e-16, 1e10)
            pdf_upper = np.clip(pdf_upper, 1e-16, 1e10)

            std_projection = self.projection_operation(std * self.beta)
            prob = prob * (
                # (mean - self.threshold) * (2 * cdf - cdf_lower - cdf_upper)
                # - std * (2 * pdf - pdf_lower - pdf_upper)
                + self.beta * std * (cdf_upper - cdf_lower)
                # + std_projection * (cdf_upper - cdf_lower)
            )

        return prob

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError()

    def projection_operation(self, std):
        return 1. / (1. + np.exp(-std))