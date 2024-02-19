import numpy as np
from GPy.models import GPRegression, GPClassification
from .gp_heteroscedastic_regression import GPHeteroscedasticRegression
from GPy.mappings import Constant
from GPy.kern import Matern52
from utils.ep_processing import ep_truncated


class GPMultipleConstraints:

    def __init__(self, dataset, models=None, constraint_num=None, classification=True, ep_modify=False, ep_loop_num=1):
        self.dataset = []
        if models is None:
            if constraint_num is None:
                print("One should design your model or constraint_number & dataset first!")
                raise ValueError()
            self.constraint_num = constraint_num
            self.classification = classification
            self.input_dim = len(dataset['OBJECTIVE'][0][0])
            self.model_obj = GPRegression(dataset['OBJECTIVE'][0], self.normalize(dataset['OBJECTIVE'][1]), noise_var=1e-6,
                                        #   mean_function=Constant(self.input_dim, output_dim=1),
                                          kernel=Matern52(self.input_dim, ARD=True))
            self.model_obj.likelihood.variance.fix()
            self.model_obj.optimize()
            self.dataset.append([dataset['OBJECTIVE'][0], dataset['OBJECTIVE'][1]])

            self.model_cons = []
            dataset_cons = []
            if classification:
                print("preparing GPClassification model for binary inputs...")
                for i in range(constraint_num):
                    con_index = 'CONSTRAINT' + str(i)
                    feasible_solutions = dataset[con_index][1][:, 0]
                    temp_con_model = GPClassification(dataset[con_index][0],
                                                      feasible_solutions.reshape([-1, 1]),
                                                      # mean_function=Constant(self.input_dim, output_dim=1),
                                                      kernel=Matern52(self.input_dim, ARD=True))
                    temp_con_model.optimize()
                    self.model_cons.append(temp_con_model)
                    dataset_cons.append([dataset[con_index][0], dataset[con_index][1]])
            else:
                print("preparing GPHeteroscedasticRegression model for mixed inputs...")
                for i in range(constraint_num):
                    con_index = 'CONSTRAINT' + str(i)
                    feasible_solutions = dataset[con_index][1][:, 0]
                    temp_con_model = GPHeteroscedasticRegression(dataset[con_index][0],
                                                                 feasible_solutions.reshape([-1, 1]),
                                                                 # mean_function=Constant(self.input_dim, output_dim=1),
                                                                 kernel=Matern52(self.input_dim, ARD=True)
                                                                 )
                    if ep_modify:
                        f_ = temp_con_model.Y
                        K_ = temp_con_model.kern.K(temp_con_model.X)
                        ep_mean, ep_sigma, logz_til, logz_EP = ep_truncated(f_,
                                                                            K_[:, 0] * 0.,
                                                                            K_,
                                                                            iteration_num=ep_loop_num,
                                                                            a=0.,
                                                                            partial=True)
                        temp_con_model['.*het_Gauss.variance'] = np.diag(ep_sigma).reshape(-1, 1)
                    else:
                        temp_con_model['.*het_Gauss.variance'] = dataset[con_index][1][:, 1][:, None]
                    temp_con_model.het_Gauss.variance.fix()
                    temp_con_model.optimize()
                    self.model_cons.append(temp_con_model)
                    dataset_cons.append([dataset[con_index][0], dataset[con_index][1]])
            self.dataset.append(dataset_cons)
        else:
            self.model_obj = models[0]
            self.model_cons = models[1]
            self.constraint_num = len(models[1])

        print("the constraint number is ", self.constraint_num)

    def update(self, new_observations):
        "Augment the dataset of the model"
        self.dataset_update(new_observations)
        self.model_obj = GPRegression(self.dataset[0][0], self.normalize(self.dataset[0][1]), noise_var=1e-6,
                                    #   mean_function=Constant(self.input_dim, output_dim=1),
                                      kernel=Matern52(self.input_dim, ARD=True)
                                      )
        self.model_obj.likelihood.variance.fix()
        self.model_obj.optimize()
        for i in range(self.constraint_num):
            feasible_solutions = self.dataset[1][i][1][:, 0]
            if self.classification:
                # GPC faces set_XY error?
                self.model_cons[i] = GPClassification(self.dataset[1][i][0],
                                                      feasible_solutions.reshape([-1, 1]),
                                                      # mean_function=Constant(self.input_dim, output_dim=1),
                                                      kernel=Matern52(self.input_dim, ARD=True))
            else:
                self.model_cons[i] = GPHeteroscedasticRegression(self.dataset[1][i][0],
                                                                 feasible_solutions.reshape([-1, 1]),
                                                                 # mean_function=Constant(self.input_dim, output_dim=1),
                                                                 kernel=Matern52(self.input_dim, ARD=True)
                                                                 )
                self.model_cons[i]['.*het_Gauss.variance'] = self.dataset[1][i][1][:, 1][:, None]
                self.model_cons[i].het_Gauss.variance.fix()

            self.model_cons[i].optimize()
        return

    def predict(self, X):
        "Get the predicted mean and std at X."
        outputs = []
        obj = self.model_obj.predict(X)
        outputs.append(obj)
        cons = []
        for i in range(self.constraint_num):
            cons.append(self.model_cons[i].predict_noiseless(X))
        outputs.append(cons)
        return outputs

    def dataset_update(self, new_observations):
        "Augment the dataset of the model"
        self.dataset[0][0] = np.vstack((self.dataset[0][0], new_observations['OBJECTIVE'][0]))
        self.dataset[0][1] = np.vstack((self.dataset[0][1], new_observations['OBJECTIVE'][1]))

        for i in range(self.constraint_num):
            con_index = 'CONSTRAINT' + str(i)
            self.dataset[1][i][0] = np.vstack((self.dataset[1][i][0], new_observations[con_index][0]))
            self.dataset[1][i][1] = np.vstack((self.dataset[1][i][1], new_observations[con_index][1]))
        return

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        m = self.model_obj.Y
        return m.min()

    def normalize(self, input_vec):
        return (input_vec - np.mean(input_vec)) / np.std(input_vec)
        # return input_vec

    def obj_min(self):
        return np.min(self.dataset[0][1])