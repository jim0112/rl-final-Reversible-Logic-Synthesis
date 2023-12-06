import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C, DQN
from transformation_based import output2gates_basic
from utils import string2index

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

register(
    id='rls-eval',
    entry_point='envs:RLS_eval'
)

def mask_fn(env:gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(seed=None, **kwargs):
    env = gym.make('rls-eval')
    env = ActionMasker(env,mask_fn)
    return env

model_path = "models/best"
env = gym.make('rls-eval')
env = ActionMasker(env,mask_fn)
#model = PPO.load(model_path, env=env)
model = MaskablePPO.load(model_path, env=env)
episodes = 1000
n = 3
list2id = string2index(n)

ours, algorithms, fail = 0, 0, 0
for i in range(episodes):
    done = False
    obs, info = env.reset()
    intlist = [list2id[''.join([str(xx) for xx in x.tolist()])] for x in obs[1].astype(int)]
    baseline = len(output2gates_basic(3, intlist))
    # print(obs)
    cnt = 0
    while not done:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        #action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(action)
        cnt += 1
    if info["MatchCnt"][-1] != 8:
        print(info["MatchCnt"][-1], cnt)
        fail += 1
    ours += cnt
    algorithms += baseline

    # modify = # of legal moves in this episode
    # matchcnt = # of matched permutations
    # print(f'Episode: {i+1}, MatchCnt: {info["MatchCnt"][-1]}, Gate: {cnt}, Baseline: {baseline}')
print(f'our steps = {ours / episodes} \nour correct rate = {(episodes - fail) / episodes}\ntransformation base = {algorithms / episodes}')
