from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from utils import SobolDesign
import numpy as np
from scipy.stats import norm


class AcquisitionMESC(AcquisitionBase):
    """
    General template to create a new GPyOPt acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients

    """
    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    def __init__(self, BOmodel, space, optimizer, cost_withGradients=None, num_samples=20, grid_size=1000, threshold=0.):
        self.optimizer = optimizer
        super(AcquisitionMESC, self).__init__(BOmodel, space, optimizer)
        self.space = space
        self.cons_num = BOmodel.constraint_num
        self.BOModel = BOmodel
        self.num_samples = num_samples
        self.threshold = threshold
        self.grid_size = grid_size
        self.min_samples = None
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('MESC acquisition does now make sense with cost at present. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, X):

        self.model_cons = self.BOModel.model_cons
        self.model_f = self.BOModel.model_obj

        # sampling the optimal and feasible value of objective function
        seed = int(np.random.uniform(1, 1e5, 1))
        sampler = SobolDesign(self.space)
        sample_points = sampler.get_samples(self.grid_size, seed)
        # sample_points = self.space.get_sample(number=self.grid_size)
        sample_points = np.concatenate((sample_points, self.model_f.X), 0)
        min_value_samples = self.sample(self.num_samples, sample_points)
        self.min_samples = min_value_samples

        # create af of mesc
        gammas_c = list()
        for i in range(self.cons_num):
            mean, var = self.model_cons[i].predict_noiseless(X)
            cons_std = np.sqrt(var)
            cons_std = np.clip(cons_std, 1e-8, 1e8) # clip below to improve numerical stability
            gammas_c.append(((self.threshold - mean) / cons_std).ravel())

        gammas_c = np.array(gammas_c)
        cdf_funcs_c = norm.cdf(gammas_c)

        Z_star = np.prod(cdf_funcs_c, axis=0)

        index_c = cdf_funcs_c == 0
        cdf_funcs_c[index_c] = 1
        inner_sum_c = gammas_c * norm.pdf(gammas_c) / cdf_funcs_c

        inner_sum_c[index_c] = gammas_c[index_c] ** 2
        inner_sum_c = np.sum(inner_sum_c, axis=0)

        mean, var = self.model_f.predict(X)
        obj_std = np.sqrt(var)
        obj_std = np.clip(obj_std, 1e-8, 1e8)  # clip below to improve numerical stability
        gamma_f = (np.squeeze(self.min_samples) - mean) / obj_std
        cdf_funcs_f = norm.cdf(gamma_f)

        Z_star = np.c_[Z_star] * cdf_funcs_f

        index_f = cdf_funcs_f == 0
        cdf_funcs_f[index_f] = 1

        gamma_f[np.isinf(gamma_f)] = 0
        inner_sum_f = gamma_f * norm.pdf(gamma_f) / cdf_funcs_f

        inner_sum_f[index_f] = gamma_f[index_f] ** 2
        inner_sum = inner_sum_f + np.c_[inner_sum_c]

        Z_star[Z_star == 1] = 1 - 1e-16

        acq = np.sum(Z_star / (2 * (1 - Z_star)) * inner_sum - np.log(1 - Z_star), axis=1)[:, None]
        # return np.abs(acq)
        return acq

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError()

    def sample(self, sample_size, X):
        """
        Return exact samples from either the objective function's minimser or its minimal value
        over the candidate set `at`.
        :param sample_size: The desired number of samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        """

        self.model_f = self.BOModel.model_obj
        samples_obj = self.model_f.posterior_samples(X, size=sample_size)  # grid * 1 * sample_num

        for k in range(self.cons_num):
            samples_cons = self.BOModel.model_cons[k].posterior_samples(X, size=sample_size)
            for i in range(len(samples_obj[:, 0, 0])):
                for j in range(self.num_samples):
                    if samples_cons[j, i] > 0:
                        samples_obj[i, 0, j] = 1e10
        thompson_samples = np.min(samples_obj, axis=0)
        return thompson_samples

    def sampling_next_point(self):
        seed = int(np.random.uniform(1, 1e5, 1))
        sampler = SobolDesign(self.space)
        sample_points = sampler.get_samples(self.grid_size, seed)
        acq = self._compute_acq(sample_points)
        best_acq = np.max(acq, axis=0)
        suggested_point = np.argmax(acq, axis=0)
        return sample_points[suggested_point], best_acq