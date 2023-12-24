import warnings
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C, DQN
from transformation_based import output2gates_basic, output2gates_bidirectional
from utils import string2index, make_test

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")

register(
    id='rls-eval',
    entry_point='envs:RLS_eval'
)

my_config = {
    # select different algorithms: PPO, A2C, DQN, MaskablePPO
    "algorithm": MaskablePPO,
    "model_path" : "models/best",

    "n" : 4,
    "eval_episodes" : 1000
}

def mask_fn(env:gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(seed=None, **kwargs):
    env = gym.make('rls-eval')
    if my_config["algorithm"] == MaskablePPO:
        env = ActionMasker(env,mask_fn)
    return env

env = make_env()
model = my_config["algorithm"].load(my_config["model_path"], env=env)

list2id = string2index(my_config["n"])
tests = make_test(my_config["n"], my_config["eval_episodes"], 1002)

ours, fail = 0, 0
b_cnt, sb_cnt = 0, 0
for i in range(my_config["eval_episodes"]):
    done = False
    obs, info = env.reset()
    obs[1] = np.array(tests[i])
    intlist = [list2id[''.join([str(xx) for xx in x.tolist()])] for x in obs[1].astype(int)]
    baseline = len(output2gates_basic(my_config["n"], intlist.copy()))
    strong_baseline = len(output2gates_bidirectional(my_config["n"], intlist.copy()))
    # print(obs)
    cnt = 0
    while not done:
        if my_config["algorithm"] == MaskablePPO:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(action)
        cnt += 1
    if info["MatchCnt"][-1] != 1 << my_config["n"]:
        #print(info["MatchCnt"][-1], cnt)
        fail += 1
    else:
        ours += cnt
        b_cnt += baseline
        sb_cnt += strong_baseline

    # modify = # of legal moves in this episode
    # matchcnt = # of matched permutations
    # print(f'Episode: {i+1}, MatchCnt: {info["MatchCnt"][-1]}, Gate: {cnt}, Baseline: {baseline}')
        
if my_config["eval_episodes"] - fail == 0 :
    print(f'the model in {my_config["model_path"]} fail all test')
else :
    print(f'our steps = {ours / (my_config["eval_episodes"] - fail)} \n',
            f'our correct rate = {(my_config["eval_episodes"] - fail) / my_config["eval_episodes"]}\n',
            f'base = {b_cnt / (my_config["eval_episodes"] - fail)} \n',
            f'bidirectional = {sb_cnt / (my_config["eval_episodes"] - fail)}')