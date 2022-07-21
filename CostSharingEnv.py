import gym
import numpy as np
from gym import spaces
from random import random

# total number of agents
n = 10


def draw_type_profile():
    def fix(x):
        return max(0, min(1, x))

    d1 = np.random.normal(0.15, 0.1, n)
    d2 = np.random.normal(0.85, 0.1, n)
    flag = np.random.uniform(0, 1, n)
    res = []
    for i in range(n):
        if flag[i] >= 0.5:
            res.append(fix(d1[i]))
        else:
            res.append(fix(d2[i]))
    return res


class CostSharingEnv(gym.Env):
    def __init__(self):
        # observation space has length 2n, each coordinate is from 0 to 1
        # for each agent, we track her already-accepted value and whether she is alive (1 means alive, 0 means out)
        self.observation_space = spaces.Box(0, 1, (2 * n,), dtype=np.float32)

        # action space is from 0 to 1.
        # we track the next_agent_to_offer, action x means increase her offer by x
        self.action_space = spaces.Box(0, 1, (1,), dtype=np.float32)

    def reset(self):
        self.type_profile = draw_type_profile()
        self.accepted = [0] * n
        self.alive = [1] * n

        # always go from left to right, circular, skipping agents who are already out
        self.next_agent_to_offer = 0

        return self._obs_construct()

    def step(self, action):
        assert 0 <= action <= 1
        # return format is: observation, reward, whether it is done, additional info dictionary
        a = self.next_agent_to_offer
        assert self.alive[a]
        if sum(self.alive) == 1:
            self.accepted[a] = 0
            self.alive[a] = 0
            return self._obs_construct(), 0, True, {}

        new_offer = self.accepted[a] + min(action, 1 - sum(self.accepted))

        if self.type_profile[a] >= new_offer:
            self.accepted[a] = new_offer
            if sum(self.accepted) >= 0.999999:  # guard against numerical error
                return self._obs_construct(), sum(self.alive), True, {}
        else:
            self.accepted[a] = 0
            self.alive[a] = 0

        self.next_agent_to_offer = None
        for i in list(range(a + 1, n)) + list(range(a)):
            if self.alive[i]:
                self.next_agent_to_offer = i
                break
        assert self.next_agent_to_offer is not None

        return (
            self._obs_construct(),
            0,
            False,
            {},
        )

    def _obs_construct(self):
        a = self.next_agent_to_offer

        # observation space always starts with the next_agent_to_offer, then followed by the other agents (sorted)
        res = [self.accepted[a], self.alive[a]]
        for current_alive, current_accept in sorted(
            zip(
                self.alive[:a] + self.alive[a + 1 :],
                self.accepted[:a] + self.accepted[a + 1 :],
            ),
            reverse=True,
        ):
            res.append(current_accept)
            res.append(current_alive)
        return np.asarray(res, dtype=np.float32)


if __name__ == "__main__":
    env = CostSharingEnv()
    env.reset()
    done = False
    while not done:
        obs, reward, done, _ = env.step(random())
        print(obs, reward, done)
