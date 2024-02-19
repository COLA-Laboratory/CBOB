import numpy as np
from .data_observation import SimEnvironmentMultiConstraints
import math


class SimEnvironmentPVD(SimEnvironmentMultiConstraints):
    def __init__(self,
                 obj_partial=True,
                 cons_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=4, classification=classification, obj_partial=obj_partial,
                         cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        xx = np.copy(x)
        step = 0.0625
        xx[:, 0] = np.round(xx[:, 0]) * step
        xx[:, 1] = np.round(xx[:, 1]) * step

        return 0.6224 * xx[:, 0] * xx[:, 2] * xx[:, 3] + \
               1.7781 * xx[:, 1] * xx[:, 2] ** 2 + 3.1661 * xx[:, 0] ** 2 * xx[:, 3] + \
               19.84 * xx[:, 0] ** 2 * xx[:, 2]

    def constraint(self, x):
        xx = np.copy(x)
        step = 0.0625
        xx[:, 0] = np.round(xx[:, 0]) * step
        xx[:, 1] = np.round(xx[:, 1]) * step

        cons1 = - xx[:, 0] + 0.0193 * xx[:, 2]
        cons2 = - xx[:, 1] + 0.00954 * xx[:, 2]
        cons3 = - math.pi * xx[:, 2] ** 2 * xx[:, 3] - 4 / 3 * math.pi * xx[:, 2] ** 3 + 1296000
        cons4 = xx[:, 3] - 240
        return [cons1, cons2, cons3, cons4]


class SimEnvironmentWBD(SimEnvironmentMultiConstraints):
    def __init__(self,
                 # constraint_num=1,
                 cons_partial=True,
                 obj_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=5, classification=classification, cons_partial=cons_partial,
                         obj_partial=obj_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        return 1.10471 * x[:, 0] ** 2 * x[:, 1] + 0.04811 * x[:, 2] * x[:, 3] * (
                14 + x[:, 1])

    def constraint(self, x):
        P = 6000
        L = 14
        tau_1 = 1 / (math.sqrt(2) * x[:, 0] * x[:, 1])
        R = np.sqrt(x[:, 1] ** 2 + (x[:, 0] + x[:, 2]) ** 2)
        tau_2 = (L + x[:, 1] / 2) * R / (
                math.sqrt(2) * x[:, 0] * x[:, 2] * (x[:, 1] ** 2 / 3 + (x[:, 0] + x[:, 2]) ** 2))

        tau = P * np.sqrt(tau_1 ** 2 + tau_2 ** 2 + 2 * tau_1 * tau_2 * x[:, 1] / R)
        sigma = P * 6 * L / (x[:, 2] ** 2 * x[:, 3])
        P_c = 64746.022 * (1 - 0.0282346 * x[:, 2]) * x[:, 2] * x[:, 3] ** 3
        delta = 2.1952 / (x[:, 2] ** 3 * x[:, 3])

        cons1 = tau - 13600
        cons2 = sigma - 30000
        cons3 = x[:, 0] - x[:, 3]
        cons4 = 6000 - P_c
        cons5 = delta - 0.25
        return [cons1, cons2, cons3, cons4, cons5]


class SimEnvironmentPistonLever(SimEnvironmentMultiConstraints):
    def __init__(self,
                 # constraint_num=1,
                 cons_partial=True,
                 obj_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=4, classification=classification, cons_partial=cons_partial,
                         obj_partial=obj_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        H = x[:, 0]
        B = x[:, 1]
        X = x[:, 2]
        D = x[:, 3]
        theta = np.pi / 4

        L1 = np.sqrt((X - B) ** 2 + H ** 2)
        L2 = np.sqrt((X * np.sin(theta) + H) ** 2 + (B - X * np.cos(theta)) ** 2)
        return 0.25 * np.pi * D ** 2 * (L2 - L1)

    def constraint(self, x):
        H = x[:, 0]
        B = x[:, 1]
        X = x[:, 2]
        D = x[:, 3]
        theta = np.pi / 4

        theta = np.pi / 4
        # Q = 1e4
        # L = 240.
        # M = 1.8 * 1e6
        # P = 1500.
        P = 10000.
        L = 240.
        M = 1.8 * 10 ** 6
        Q = 1500

        R = np.abs(- X * (X * np.sin(theta) + H) + H * (B - X * np.cos(theta))) / np.sqrt((X - B) ** 2 + H ** 2)
        F = np.pi * P * D ** 2 / 4
        L1 = np.sqrt((X - B) ** 2 + H ** 2)
        L2 = np.sqrt((X * np.sin(theta) + H) ** 2 + (B - X * np.cos(theta)) ** 2)

        cons1 = Q * L * np.cos(theta) - R * F
        cons2 = Q * (L - x[:, 3]) - M
        cons3 = 1.2 * (L2 - L1) - L1
        cons4 = x[:, 2] / 2 - x[:, 1]

        return [cons1, cons2, cons3, cons4]


class SimEnvironmentTensionCompressionString(SimEnvironmentMultiConstraints):
    def __init__(self,
                 # constraint_num=1,
                 cons_partial=True,
                 obj_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ):
        super().__init__(constraint_num=4, classification=classification, cons_partial=cons_partial,
                         obj_partial=obj_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def objective(self, x):
        return x[:, 0] ** 2 * x[:, 1] * (x[:, 2] + 2.)

    def constraint(self, x):

        cons1 = 1 - x[:, 1] ** 3 * x[:, 2] / (71785 * x[:, 0] ** 4)
        cons2 = (4 * x[:, 1] ** 2 - x[:, 0] * x[:, 1]) / (12566 * x[:, 0] ** 3 * (x[:, 1] - x[:, 0])) + 1 / (5108 * x[:, 0] ** 2) - 1.
        cons3 = 1 - 140.45 * x[:, 0] / (x[:, 2] * x[:, 1] ** 2)
        cons4 = (x[:, 0] + x[:, 1]) / 1.5 - 1.
        return [cons1, cons2, cons3, cons4]

