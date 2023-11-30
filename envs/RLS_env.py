from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
import random
from copy import deepcopy

import numpy as np

class RLS(gym.Env):
    def __init__(self):
        # inp shape = {2^n, n}
        self.n = 3
        self.base = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(0, 1, (3, 2 ** self.n, self.n), dtype=int)
        # can be changed to whatever u want
        self.step_penalty = -1
        self.illegalPenalty = -5
        self.illegalMax = 20
        self.max_step = 200

        self.actiondict = {0: [0,0,2], 1: [0,1,2], 2: [0,2,2], 3: [1,0,2], 4: [1,1,2], 5: [1,2,2], 6: [2,0,2], 
                           7: [2,1,2], 8: [0,2,1], 9: [1,2,1], 10: [2,0,1], 11: [2,1,1], 12: [2,2,1], 13: [0,2,0], 
                           14: [1,2,0], 15:[2,0,0], 16: [2,1,0], 17: [2,2,0]}
        self.reset()

    def hammingDistanceMetric(self, state):
        old_distance, new_distance = 0, 0
        for ori, new, out in zip(self.state.flatten(), state.flatten(), self.out.flatten()):
            if ori != out:
                old_distance += 1
            if new != out:
                new_distance += 1
        o = new_distance - old_distance
        return o

    def permutationMetric(self, state):
        old_match, new_match = 0, 0
        for ori, new, out in zip(self.state, state, self.out):
            if (ori == out).all():
                old_match += 1
            if (new == out).all():
                new_match += 1
        return new_match - old_match

    def mismatchMetric(self, state):
        import math
        lc, mc = 0, 0
        eps = 1e-6
        for new, out in zip(state, self.out):
            if new[-1] != out[-1]:
                lc += 1
            if new[0] != out[0]:
                mc += 1
        n = state.shape[0]
        o = math.log2(n / (lc + eps)) + math.log2(n / (mc + eps))
        # the higher the better
        return o

    def LSBMetric(self, state):
        curr = 1
        gamma = 1.2
        reward = 0
        data = list(zip(self.out, state))
        for out, new in sorted(data, key=lambda x: int(''.join(map(str, x[0])), 2)):
            # print(out)
            if (new == out).all():
                reward += curr
                curr *= gamma
            else: break
        return reward

    def illegalMove(self, state):
        for ori, new, out in zip(self.state, state, self.out):
            # if already matched, then any change on it is illegal
            if (ori == out).all():
                if not (new == out).all():
                    return True
            else:
                break
        return False

    def visualize(self):
        print(f'Initial State:\n {np.transpose(self.initial_state)}')
        print(f'Ending State:\n {np.transpose(self.state)}')
        print(f'Target State:\n {np.transpose(self.out)}')
        print(f'Derived Circuit: {len(self.gateTrace)} levels')
        for ele in np.transpose(self.gateTrace):
            print(ele)
        print()
    
    def trace(self, action):
        # MetaData: gates, match count
        self.gateTrace.append(action)
        cnt = 0
        for new, out in zip(self.state, self.out):
            if (new == out).all():
                cnt += 1
        self.matchCntTrace.append(cnt)


    def step(self, action):
        # do the gate operation, done
        done = False
        new_state = []
        control = []
        action = self.actiondict[action]
        # 0: no, 1: control, 2: not
        for i, a in enumerate(action):
            if a == 1:
                control.append(i)

        for s in self.state:
            flip = 1
            for c in control:
                flip *= s[c]
            tmp = []
            for i, bit in enumerate(s):
                if flip and action[i] == 2:
                    tmp.append(0 if bit == 1 else 1)
                else:
                    tmp.append(bit)
            new_state.append(tmp)
        new_state = np.array(new_state)

        reward = self.hammingDistanceMetric(new_state)
        # reward = self.permutationMetric(new_state)

        if self.illegalMove(new_state) or (new_state == self.pre).all():
            reward = self.illegalPenalty
            self.illegalCnt += 1
        else:
            self.pre = self.state
            self.state = new_state
            self.modify += 1
        reward += self.step_penalty
        
        self.step_cnt += 1
        self.trace(action)

        if self.step_cnt == self.max_step or (self.state == self.out).all():
            done = True
            # self.visualize()
        elif self.illegalCnt == self.illegalMax:
            done = True
            # self.visualize()
        
        info = {'MatchCnt': self.matchCntTrace,
                'GateTrace': self.gateTrace,
                'Modify': self.modify,
                'Initial_State': self.initial_state,
                'End_State': self.state,
        }

        return np.stack([self.pre, self.state, self.out], axis=0), reward, done, False, info

    def reset(self, seed=None):
        self.state = np.array(random.sample(self.base, len(self.base)))
        self.pre = np.zeros(self.state.shape) - 1
        self.out = np.array(self.base)
        self.step_cnt = 0
        self.illegalCnt = 0
        self.modify = 0

        self.initial_state = self.state
        
        # self.used.clear()
        self.gateTrace = []
        self.matchCntTrace = []
        

        return np.stack([self.pre, self.state, self.out], axis=0), {}

    def render(self):
        # no use
        pass
