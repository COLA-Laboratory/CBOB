import numpy as np


class SimEnvironmentMultiConstraints:
    """
    define the observation environment of objective and multiple constraints, generate the data while observing
    """

    def __init__(self, constraint_num=None, decoupled=True, obj_partial=True, cons_partial=True, classification=False,
                 infeasible_value=None, feasible_value=None, virtual_value=2.0):
        if constraint_num is None:
            print("Missing the number of constraint!")
            raise ValueError()
        if classification:
            if infeasible_value is None or feasible_value is None:
                print("should determine the cons value of an infeasible point!")
                raise ValueError()
            self._infeasible_value = infeasible_value
            self._feasible_value = feasible_value
        else:
            self._virtual_value = virtual_value
        self.classification = classification
        self.OBJECTIVE = "OBJECTIVE"
        self.CONSTRAINT = []
        self.decoupled = decoupled
        self.obj_partial = obj_partial
        self.cons_partial = cons_partial

        for i in range(constraint_num):
            self.CONSTRAINT.append("CONSTRAINT" + str(i))

    def observer(self, query_point, f_print=True):
        # define the objective
        if self.decoupled:
            cons = self.constraint(query_point)
            # obj = self.objective(input_data)[cons <= 0]
            # feasible_data = input_data[cons <= 0]
            if self.obj_partial:
                one = np.ones(shape=cons[0].shape)
                zero = np.zeros(shape=cons[0].shape)
                b = one
                for i in range(len(cons)):
                    b = np.where(cons[i] <= 0, b, zero)
                feasible_query_point = query_point[b > 0]
            else:
                feasible_query_point = query_point

            obj = self.objective(feasible_query_point).reshape([-1, 1])

            # define the constraints
            cons = self.mask_cons(cons)
            if f_print:
                print("query points = ", query_point)
                print("searching obj = ", obj)
                print("searching cons = ", cons)

            constraints = {}
            for i in range(len(self.CONSTRAINT)):
                constraints = {**constraints, **{self.CONSTRAINT[i]: (query_point, cons[i])}}
            return {**{self.OBJECTIVE: (feasible_query_point, obj)}, **constraints}
            # return constraints
        else:
            raise NotImplementedError()

    def mask_cons(self, y):
        assert len(y) == len(self.CONSTRAINT)
        output_y = []
        # noise_var = y * 0.0 + 1e-6
        for i in range(len(y)):
            yi = y[i]
            noise_var_i = yi * 0.0 + 1e-6
            for j in range(len(yi)):
                if not self.classification:
                    if yi[j] > 0:
                        if not self.cons_partial:
                            yi[j] = np.log(1 + yi[j])
                            noise_var_i[j] = yi[j] * 0.0 + 1e-6
                        else:
                            yi[j] = np.log(1 + self._virtual_value)
                            noise_var_i[j] = (0.44 * np.abs(yi[j])) ** 2 + 1e-6
                    else:
                        yi[j] = - np.log(- yi[j] + 1)
                        noise_var_i[j] = yi[j] * 0.0 + 1e-6
                else:
                    if yi[j] > 0:
                        yi[j] = self._infeasible_value
                    else:
                        yi[j] = self._feasible_value
            yi = yi.reshape(-1, 1)
            noise_var_i = noise_var_i.reshape(-1, 1)
            y_data_i = np.hstack((yi, noise_var_i))
            output_y.append(y_data_i)
        return output_y

    def objective(self, x):
        """
        define the objective function
        """
        return x[:, 0] + 2 * x[:, 2] * x[:, 1]

    def constraint(self, x):
        """
        define the constraint function
        """
        cons1 = x[:, 0] - 2 * x[:, 2] * x[:, 1]
        cons2 = - 4 * x[:, 0] - 0.5 * x[:, 2] + 0.5 * x[:, 1]
        return [cons1, cons2]


if __name__ == '__main__':
    X = np.random.random(size=(10, 3))
    envs = SimEnvironmentMultiConstraints(constraint_num=2, classification=True, feasible_value=1., infeasible_value=0.)
    observs = envs.observer(X)
