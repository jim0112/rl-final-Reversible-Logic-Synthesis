import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C, DQN
from transformation_based import output2gates

register(
    id='rls-eval',
    entry_point='envs:RLS_eval'
)

model_path = "models/best"
env = gym.make('rls-eval')
model = PPO.load(model_path, env=env)
episodes = 1000
list2id = {'000': 0, '001':1, '010':2, '011':3, '100':4, '101':5, '110': 6, '111':7}

ours, algorithms = 0, 0
for i in range(episodes):
    done = False
    obs, info = env.reset()
    intlist = [list2id[''.join([str(xx) for xx in x.tolist()])] for x in obs[1].astype(int)]
    baseline = len(output2gates(3, intlist))
    # print(obs)
    cnt = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(action)
        cnt += 1
    if info["MatchCnt"][-1] != 8:
        print(info["MatchCnt"][-1], cnt)
    ours += cnt
    algorithms += baseline

    # modify = # of legal moves in this episode
    # matchcnt = # of matched permutations
    # print(f'Episode: {i+1}, MatchCnt: {info["MatchCnt"][-1]}, Gate: {cnt}, Baseline: {baseline}')
print(f'our steps = {ours / episodes} \n transformation base = {algorithms / episodes}')