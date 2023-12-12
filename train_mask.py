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

#import wandb
#from wandb.integration.sb3 import WandbCallback

warnings.filterwarnings("ignore")

register(
    id='rls-v0',
    entry_point='envs:RLS_eval'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    #"algorithm": PPO,
    "algorithm": MaskablePPO, #Todo: select different algorithms: A2C, DQN
    #"policy_network": "MlpPolicy",
    "policy_network": MaskableActorCriticPolicy,

    "epoch_num": 500,
    "timesteps_per_epoch": 10000,
    "eval_episode_num": 10,
}
def mask_fn(env:gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(seed=None, **kwargs):
    env = gym.make('rls-v0')
    env = ActionMasker(env,mask_fn)
    return env

def train(env, model, config):

    current_best = -10000000
    matched_best = 0
    eval_epochs = 100
    tests = make_test(4, eval_epochs, 5487) # select a particular seed for evaluation

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            
#            callback=WandbCallback(
#                gradient_save_freq=100,
#                verbose=2,
#            ),
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
                # if don't apply action mask during eval, in the eval action wouldn't be masked
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
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
#        wandb.log(
#            {"avg_matched": matched/10,
#            "avg_score": total_score/10}
#        )
        ### Save best model
        if current_best <= total_score and matched >= matched_best:
            print(f"Saving Model with avg score {total_score / eval_epochs} and avg matchCnt {matched / eval_epochs}")
            current_best = total_score
            matched_best = matched
            save_path = 'models'
            model.save(f"{save_path}/best")

        print("---------------")

        print(current_best / eval_epochs)
        print(matched_best / eval_epochs)

if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    
#    run = wandb.init(
#        project="rl_final_1205_v5",
#        config=my_config,
#        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#        #id=my_config["run_id"]
#        id = "3_action_mask"
#    )
    
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
