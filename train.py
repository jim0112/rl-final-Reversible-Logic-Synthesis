import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from utils import make_test

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

warnings.filterwarnings("ignore")

register(
    id='rls-v0',
    entry_point='envs:RLS_eval'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",
    # select different algorithms: PPO, A2C, DQN, MaskablePPO
    "algorithm": MaskablePPO,
    "model_path" : "models/best",
    "policy_network": "MlpPolicy",

    "n":3,

    "epoch_num": 500,
    "timesteps_per_epoch": 500,
    "eval_episode_num": 100,
}
def mask_fn(env:gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(seed=None, **kwargs):
    env = gym.make('rls-v0')
    if my_config["algorithm"] == MaskablePPO:
        env = ActionMasker(env,mask_fn)
    return env

def train(env, model, config):

    current_best = -10000000
    matched_best = 0
    eval_epochs = config["eval_episode_num"]
    tests = make_test(config["n"], eval_epochs, 5487) # select a particular seed for evaluation

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
        )

        ### Evaluation
        print("Epoch: ", epoch)
        total_score = 0
        matched = 0
        for i in range(eval_epochs):
            done = False
            score = 0
            obs = env.reset()
            obs[0][1] = tests[i]
            cnt = 0
            temp = []
            while not done:
                if config["algorithm"] == MaskablePPO:
                    action_masks = get_action_masks(env)
                    # if don't apply action mask during eval, in the eval action wouldn't be masked
                    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, r, done, info = env.step(action)
                score += r[0]
                temp.append(r[0])
                cnt += 1
            total_score += score
            matched += info[0]["MatchCnt"][-1]
            # modify = # of legal moves in this episode
            # matchcnt = # of matched permutations
            # print(temp)
            # print(f'Episode: {i+1}, Score: {score}, Gate: {cnt}, MatchCnt: {info[0]["MatchCnt"][-1]}, Modify: {info[0]["Modify"]}, \n Initial State: \n {np.transpose(info[0]["Initial_State"])}, \n End State: \n {np.transpose(info[0]["End_State"])}')
            # gates = info[0]["GateTrace"]
            # for ele in np.transpose(gates):
            #     print(ele)
            
        print(f'the average score is {total_score / eval_epochs}')
        ### Save best model
        if current_best <= total_score and matched >= matched_best:
            print(f"Saving Model with avg score {total_score / eval_epochs} and avg matchCnt {matched / eval_epochs}")
            current_best = total_score
            matched_best = matched
            model.save(config["model_path"])

        print("---------------")

        print(current_best / eval_epochs)
        print(matched_best / eval_epochs)

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env,
        verbose=0,
        learning_rate=0.0003,
        # tensorboard_log=my_config["run_id"]
    )
    train(env, model, my_config)