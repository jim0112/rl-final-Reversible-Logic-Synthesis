import gymnasium as gym
from gymnasium import spaces
import random
from utils import make_actions, make_base, base_mask

import numpy as np

class RLS_eval(gym.Env):
    def __init__(self):
        # inp shape = {2^n, n}
        self.n = 4 # Todo: test the limit of n
        self.base = make_base(self.n)
        self.actiondict = make_actions(self.n)
        self.action_space = spaces.Discrete(len(self.actiondict))
        self.observation_space = spaces.Box(0, 1, (3, 2 ** self.n, self.n), dtype=int)
        # can be changed to whatever u want
        self.step_penalty = -1
        self.illegalPenalty = -5
        self.illegalMax = 20
        self.max_step = 200
        self.base_mask = base_mask(self.base,self.actiondict)
        self.last_action = 0
        self.reset()

    def hammingDistanceMetric(self, state):
        old_distance, new_distance = 0, 0
        for ori, new, out in zip(self.state.flatten(), state.flatten(), self.out.flatten()):
            if ori != out:
                old_distance += 1
            if new != out:
                new_distance += 1
        o = new_distance - old_distance
        # as lower distance is better, giving -o
        return -o

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
    
    def valid_action_mask(self):
        cur = np.ones((1,len(self.actiondict)))
        cur[0][self.last_action] = 0

        max_idx = -1
        # apply original transformed based mask
        for ori, out in zip(self.state, self.out):
            max_idx += 1
            if (ori == out).all():
                cur = np.logical_and(cur, self.base_mask[max_idx])
            else :
                return cur
        return cur

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
        self.last_action = action
        action = self.actiondict[action.item()]
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

        # remove penalty
        """
        if self.illegalMove(new_state) or (new_state == self.pre).all():
            reward = self.illegalPenalty
            self.illegalCnt += 1
        else:
            self.pre = self.state
            self.state = new_state
            self.modify += 1
        """

        self.pre = self.state
        self.state = new_state
        self.modify += 1

        reward += self.step_penalty
        self.step_cnt += 1
        self.trace(action)

        if self.step_cnt == self.max_step or (self.state == self.out).all():
            done = True
            # self.visualize()
        # elif self.illegalCnt == self.illegalMax:
        #     done = True
        #     # self.visualize()
        
        info = {'MatchCnt': self.matchCntTrace,
                'GateTrace': self.gateTrace,
                'Modify': self.modify,
                'Initial_State': self.initial_state,
                'End_State': self.state,
        }

        return np.stack([self.pre, self.state, self.out], axis=0).astype(int), reward, done, False, info

    def reset(self, **kwargs):
        # add this to handle strange error that RLS.reset() doesn't eat options argument
        kwargs.pop('options', None)
        self.state = np.array(random.sample(self.base, len(self.base)))
        self.pre = np.zeros(self.state.shape) - 1 #Todo: more pre_state
        self.out = np.array(self.base)
        self.step_cnt = 0
        self.illegalCnt = 0
        self.modify = 0

        self.initial_state = self.state
        
        # self.used.clear()
        self.gateTrace = []
        self.matchCntTrace = []
        

        return np.stack([self.pre, self.state, self.out], axis=0).astype(int), {}

    def render(self):
        # no use
        pass
