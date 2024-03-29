import numpy as np

from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.experiment_design.base import ExperimentDesign
from GPyOpt.experiment_design.random_design import RandomDesign


class SobolDesign(ExperimentDesign):
    """
    Sobol experiment design.
    Uses random design for non-continuous variables, and Sobol sequence for continuous ones
    """
    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Sampling with constraints is not allowed by Sobol design')
        super(SobolDesign, self).__init__(space)

    def get_samples(self, init_points_count, seed=None):
        if seed is None:
            seed = int(np.random.uniform(1, 2**16))
        samples = np.empty((init_points_count, self.space.dimensionality))

        # Use random design to fill non-continuous variables
        random_design = RandomDesign(self.space)
        random_design.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            bounds = self.space.get_continuous_bounds()
            lower_bound = np.asarray(bounds)[:,0].reshape(1,len(bounds))
            upper_bound = np.asarray(bounds)[:,1].reshape(1,len(bounds))
            diff = upper_bound-lower_bound

            from sobol_seq import i4_sobol_generate
            X_design = np.dot(i4_sobol_generate(len(self.space.get_continuous_bounds()),init_points_count,seed),np.diag(diff.flatten()))[None,:] + lower_bound
            samples[:, self.space.get_continuous_dims()] = X_design

        return samples