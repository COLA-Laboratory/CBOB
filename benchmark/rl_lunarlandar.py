import numpy as np
import gym
from .data_observation import SimEnvironmentMultiConstraints


class SimEnvironmentLunar(SimEnvironmentMultiConstraints):
    def __init__(self,
                 cons_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 ctrl_reward_threshold=40.):
        self._ctrl_reward_threshold = ctrl_reward_threshold
        super().__init__(constraint_num=1, classification=classification, cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def observer(self, query_point):
        # define the objective
        fwd_r = np.zeros(len(query_point[:, 0]))
        ctrl_r = np.zeros(len(query_point[:, 0]))
        for i in range(len(query_point[:, 0])):
            fwd_r[i], ctrl_r[i] = lunar_lander_simulation(query_point[i, :], print_reward=True)
        # define the constraints

        feasible_query_point = query_point[ctrl_r <= self._ctrl_reward_threshold]
        obj = fwd_r[ctrl_r <= self._ctrl_reward_threshold][:, None]
        y = ctrl_r - self._ctrl_reward_threshold
        noise_var = ctrl_r * 0.0 + 1e-6
        for i in range(len(y)):
            if y[i] > 0:
                y[i] = np.log(1 + y[i])
            else:
                y[i] = - np.log(- y[i] + 1)
        y = y.reshape(-1, 1)
        noise_var = noise_var.reshape(-1, 1)
        cons = np.hstack((y, noise_var))

        print("query points = ", query_point)
        print("searching obj = ", obj)
        print("searching cons = ", cons)

        constraints = {}
        constraints = {**constraints, **{self.CONSTRAINT[0]: (query_point, cons)}}
        return {**{self.OBJECTIVE: (feasible_query_point, obj)}, **constraints}


def lunar_lander_simulation(w, is_continuous=False, print_reward=False, seed=1, steps_limit=1000, timeout_reward=100):
    total_reward = 0.
    energy_cost = 0.
    task_done = False
    steps = 0
    env_name = "LunarLander-v2"
    env = gym.make(env_name, continuous=is_continuous)
    s = env.reset(seed=seed)
    while True:
        if steps > steps_limit:
            total_reward -= timeout_reward
            break
        a = heuristic_controller(s, w)
        # calculate the energy cost
        m_power = 0
        if (is_continuous and a[0] > 0.0) or (
                not is_continuous and a == 2
        ):
            # Main engine
            if is_continuous:
                m_power = (np.clip(a[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
        s_power = 0.0
        if (is_continuous and np.abs(a[1]) > 0.5) or (
                not is_continuous and a in [1, 3]
        ):
            # Orientation engines
            if is_continuous:
                direction = np.sign(a[1])
                s_power = np.clip(np.abs(a[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = a - 2
                s_power = 1.0
        energy_cost += (
            m_power
        )
        energy_cost += s_power * 0.1

        s, r, done, info = env.step(a)
        # dismiss the reward of cost
        r += (
                m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        r += s_power * 0.03
        total_reward += r
        steps += 1
        if done:
            task_done = True
            break
    if print_reward:
        if task_done:
            print("one experiment is done. The energy cost is: ", energy_cost)
        else:
            print("one experiment is failed. The energy cost is: ", energy_cost)
        print(f"Total reward: {total_reward}")
    return -total_reward, energy_cost


def heuristic_controller(s, w, is_continuous=False):
    # w is the array of controller parameters of shape (1, 12)
    angle_target = s[0] * w[0] + s[2] * w[1]
    if angle_target > w[2]:
        angle_target = w[2]
    if angle_target < w[-2]:
        angle_target = -w[2]
    hover_target = w[3] * np.abs(s[0])
    angle_todo = (angle_target - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_target - s[1]) * w[6] - (s[3]) * w[7]
    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]
    if is_continuous:
        a = np.array([hover_todo * 20 - 1, angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < - w[11]:
            a = 3
        elif angle_todo > + w[11]:
            a = 1
    return a
