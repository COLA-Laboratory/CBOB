from GPyOpt.core.evaluators.base import EvaluatorBase


class Sequential(EvaluatorBase):
    """
    Class for standard Sequential Bayesian optimization methods.

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: it is 1 by default since this class is only used for sequential methods.
    """

    def __init__(self, acquisition, batch_size=1, is_sampling=False, updating=False):
        self.is_sampling = is_sampling
        self.updating = updating

        super(Sequential, self).__init__(acquisition, batch_size)

    def compute_batch(self, duplicate_manager=None,context_manager=None):
        """
        Selects the new location to evaluate the objective.
        """
        x, acq_value = self.acquisition.optimize(duplicate_manager=duplicate_manager)
        return x, acq_value

    def next_point(self, duplicate_manager=None, context_manager=None, grid_size=1000):
        if not self.is_sampling:
            if self.updating:
                self.acquisition.update_balance_rate()
            return self.compute_batch(duplicate_manager, context_manager=context_manager)
        else:
            return self.acquisition.sampling_next_point()
