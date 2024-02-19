from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from utils import SobolDesign
import numpy as np


class AcquisitionConstrainedThompson(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, BOmodel, space, optimizer, cost_withGradients=None, grid_size=1000, threshold=0., classification=False):
        self.optimizer = optimizer
        super(AcquisitionConstrainedThompson, self).__init__(BOmodel, space, optimizer)
        self.space = space
        self.cons_num = BOmodel.constraint_num
        self.BOModel = BOmodel
        self.threshold = threshold
        self.grid_size = grid_size
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('Thompson acquisition does now make sense with cost at present. Cost set to constant.')
            raise ValueError()

    def sampling_next_point(self):
        """
        Return exact samples from either the objective function's minimser or its minimal value
        over the candidate set `at`.
        :param sample_size: The desired number of samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        """
        seed = int(np.random.uniform(1, 1e5, 1))
        sampler = SobolDesign(self.space)
        sample_points = sampler.get_samples(self.grid_size, seed)
        sample_points = np.concatenate((sample_points, self.BOModel.model_obj.X), 0)

        self.model_f = self.BOModel.model_obj
        samples_obj = self.model_f.posterior_samples(sample_points, size=1).flatten()  # grid * 1 * sample_num

        for k in range(self.cons_num):
            samples_cons = self.BOModel.model_cons[k].posterior_samples(sample_points, size=1).flatten()
            for i in range(len(samples_obj)):
                if samples_cons[i] > 0:
                    samples_obj[i] = 1e10
        thompson_samples = np.min(samples_obj, axis=0)
        suggested_point = np.argmin(samples_obj, axis=0)
        return sample_points[suggested_point][None, :], thompson_samples

    def _compute_acq(self, x):
        raise NotImplementedError()

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError()
