from envs.RLS_env import RLS
import numpy as np
import itertools

inp = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
out = [[1,1,0],[0,0,0],[1,1,1],[0,1,1],[1,0,0],[0,1,0],[1,0,1],[0,0,1]]
env = RLS()
episodes = 1
for i in range(episodes):
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        print(action)
        n_state, r, done, t, _ = env.step(action)
        #print(n_state, r)
        score += r
    print(f'Episode: {i+1}, Score: {score}')
    state = env.reset()
