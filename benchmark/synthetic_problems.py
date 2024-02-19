import numpy as np
from .data_observation import SimEnvironmentMultiConstraints
import math


class SimEnvironmentAckley(SimEnvironmentMultiConstraints):
    def __init__(self,
                 obj_partial=True,
                 cons_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=1, classification=classification, obj_partial=obj_partial,
                         cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        a = 20
        b = 0.2
        c = 2 * 3.14159265
        d = 10
        f_square = 0. * x[:, 0]
        f_cos = 0. * x[:, 0]
        if len(x) > 0:
            for i in range(len(x[0, :])):
                f_square += x[:, i] ** 2
                f_cos += np.cos(c * x[:, i])
        return - a * np.exp(-b * np.sqrt(1 / d * f_square)) - np.exp(1 / d * f_cos) + a + 2.71828183

    def constraint(self, x):
        c1 = 0. * x[:, 0]
        for i in range(len(x[0, :])):
            c1 += x[:, i]
        cons1 = c1
        # cons2 = np.linalg.norm(x, axis=1) - 5.
        return [cons1]


class SimEnvironmentStyblinski(SimEnvironmentMultiConstraints):
    def __init__(self,
                 obj_partial=True,
                 cons_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=1, classification=classification, obj_partial=obj_partial,
                         cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        f = 0. * x[:, 0]
        if len(x) > 0:
            for i in range(len(x[0, :])):
                f += 0.5 * (x[:, i] ** 4 - 16 * x[:, i] ** 2 + 5 * x[:, i])
        return f

    def constraint(self, x):
        c1 = 0. * x[:, 0]
        for i in range(len(x[0, :])):
            c1 += np.sin(x[:, i])
        cons1 = c1
        # cons2 = np.linalg.norm(x, axis=1) - 5.
        return [cons1]



class SimEnvironmentKeane(SimEnvironmentMultiConstraints):
    def __init__(self,
                 obj_partial=True,
                 cons_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=2, classification=classification, obj_partial=obj_partial,
                         cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        cos_sum = 0. * x[:, 0]
        cos_mul = 0. * x[:, 0] + 1.
        sqr_sum = 0. * x[:, 0]
        if len(x) > 0:
            for i in range(len(x[0, :])):
                cos_sum += np.cos(x[:, i]) ** 4
                cos_mul *= np.cos(x[:, i]) ** 2
                sqr_sum += (i + 1) * x[:, i] ** 2
        return - np.abs((cos_sum - 2 * cos_mul) / np.sqrt(sqr_sum))

    def constraint(self, x):
        x_sum = 0. * x[:, 0]
        x_mul = 0. * x[:, 0] + 1.
        len_m = len(x[0])
        if len(x) > 0:
            len_m = len(x[0])
            for i in range(len(x[0, :])):
                x_sum += x[:, i]
                x_mul *= x[:, i]
        cons1 = 0.75 - x_mul
        cons2 = x_sum - 7.5 * len_m
        return [cons1, cons2]


class SimEnvironmentIllustration(SimEnvironmentMultiConstraints):
    def __init__(self,
                 obj_partial=True,
                 cons_partial=False,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=1, classification=classification, obj_partial=obj_partial,
                         cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        return (np.sin(x[:, 0]) * np.sin(2 * x[:, 0]) - np.cos(5 * x[:, 0]))

    def constraint(self, x):
        return [-(np.sin(x[:, 0]) * np.sin(2 * x[:, 0]) - np.cos(5 * x[:, 0]))]
