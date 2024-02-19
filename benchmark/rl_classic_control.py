import numpy as np
import gym
# from .data_observation import SimEnvironmentMultiConstraints
#
#
# class SimEnvironmentSwimmer(SimEnvironmentMultiConstraints):
#     def __init__(self,
#                  # constraint_num=1,
#                  cons_partial=True,
#                  classification=False,
#                  infeasible_value=None,
#                  feasible_value=None,
#                  virtual_value=2.0,
#                  ctrl_reward_threshold=1.2):
#         self._ctrl_reward_threshold = ctrl_reward_threshold
#         super().__init__(constraint_num=1, classification=classification, cons_partial=cons_partial,
#                          virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)
#
#     def observer(self, query_point):
#         # define the objective
#         swimmer_test = Pendulum()
#         fwd_r = np.zeros(len(query_point[:, 0]))
#         ctrl_r = np.zeros(len(query_point[:, 0]))
#         for i in range(len(query_point[:, 0])):
#             fwd_r[i], ctrl_r[i] = swimmer_test(query_point[i, :])
#         # define the constraints
#
#         feasible_query_point = query_point[ctrl_r <= self._ctrl_reward_threshold]
#         obj = fwd_r[ctrl_r <= self._ctrl_reward_threshold][:, None]
#         y = ctrl_r - self._ctrl_reward_threshold
#         noise_var = ctrl_r * 0.0 + 1e-6
#         for i in range(len(y)):
#             if y[i] > 0:
#                 y[i] = np.log(1 + y[i])
#             else:
#                 y[i] = - np.log(- y[i] + 1)
#         y = y.reshape(-1, 1)
#         noise_var = noise_var.reshape(-1, 1)
#         cons = np.hstack((y, noise_var))
#
#         print("query points = ", query_point)
#         print("searching obj = ", obj)
#         print("searching cons = ", cons)
#
#         constraints = {}
#         constraints = {**constraints, **{self.CONSTRAINT[0]: (query_point, cons)}}
#         return {**{self.OBJECTIVE: (feasible_query_point, obj)}, **constraints}


class InvertedPendulum:

    def __init__(self):
        self.policy_shape = (1, 4)
        self.mean = 0
        self.std = 0.1
        self.dims = 4
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make('InvertedPendulum-v4')
        self.num_rollouts = 5

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 20
        self.leaf_size = 10
        self.kernel_type = "poly"
        self.gamma_type = "scale"
        self.ninits = 40
        # print("===========initialization===========")
        # print("mean:", self.mean)
        # print("std:", self.std)
        # print("dims:", self.dims)
        # print("policy:", self.policy_shape)

        self.render = False

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_shape)
        returnsr = []
        observations = []
        actions = []
        total_step = 0

        for i in range(self.num_rollouts):
            obs = self.env.reset(seed=i)
            done = False
            totalr = 0.
            totalc = 0.

            while not done:
                action = np.dot(M, obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, info = self.env.step(action)
                totalr += r
                if self.render:
                    self.env.render()
                total_step += 1
            returnsr.append(totalr)

        return np.mean(returnsr) * -1, total_step / self.num_rollouts


class DoubleInvertedPendulum:

    def __init__(self):
        self.policy_shape = (1, 11)
        self.mean = 0
        self.std = 0.1
        self.dims = 11
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make("InvertedDoublePendulum-v4")
        self.num_rollouts = 5

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 20
        self.leaf_size = 10
        self.kernel_type = "poly"
        self.gamma_type = "scale"
        self.ninits = 40
        # print("===========initialization===========")
        # print("mean:", self.mean)
        # print("std:", self.std)
        # print("dims:", self.dims)
        # print("policy:", self.policy_shape)

        self.render = False

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_shape)
        returnsr = []
        returnsc = []
        observations = []
        actions = []
        total_step = 0

        for i in range(self.num_rollouts):
            obs = self.env.reset(seed=i)
            done = False
            totalr = 0.
            totalc = 0.

            while not done:
                action = np.dot(M, obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, info = self.env.step(action)
                totalr += r
                if self.render:
                    self.env.render()
                total_step += 1
            returnsr.append(totalr)
            # returnsc.append(totalc)
        return np.mean(returnsr) * -1, total_step / self.num_rollouts


class Reacher:

    def __init__(self):
        self.policy_shape = (2, 11)
        self.mean = 0
        self.std = 0.1
        self.dims = 22
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.counter = 0
        self.env = gym.make("Reacher-v4")
        self.num_rollouts = 5
        self.render = False

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_shape)
        returnsr = []
        returnsc = []
        observations = []
        actions = []
        total_step = 0

        for i in range(self.num_rollouts):
            obs = self.env.reset(seed=i)
            done = False
            totalr = 0.
            totalc = 0.

            while not done:
                action = np.dot(M, obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, info = self.env.step(action)
                totalr += info['reward_dist']
                totalc += info['reward_ctrl']
                if self.render:
                    self.env.render()
                total_step += 1
            returnsr.append(totalr)
            returnsc.append(totalc)

        return np.mean(returnsr) * -1, - np.mean(returnsc)


if __name__ == '__main__':
    # num = 10
    # pendulum_test = InvertedPendulum()
    # x = np.random.uniform(-1, 1, (4, num))
    # for i in range(num):
    #     r = pendulum_test(x[:, i])
    #     print(r)

    # num = 100
    # pendulum_test = DoubleInvertedPendulum()
    # x = np.random.uniform(-1, 1, (11, num))
    # for i in range(num):
    #     r = pendulum_test(x[:, i])
    #     print(r)

    num = 10
    pendulum_test = Reacher()
    x = np.random.uniform(-1, 1, (22, num))
    for i in range(num):
        r = pendulum_test(x[:, i])
        print(r)