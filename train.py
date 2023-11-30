import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

warnings.filterwarnings("ignore")

register(
    id='rls-v0',
    entry_point='envs:RLS'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": PPO,
    "policy_network": "MlpPolicy",

    "epoch_num": 100,
    "timesteps_per_epoch": 5000,
    "eval_episode_num": 10,
}

def make_env():
    env = gym.make('rls-v0')
    return env

def train(env, model, config):

    current_best = -10000000

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        print("Epoch: ", epoch)
        total_score = 0
        for i in range(10):
            done = False
            score = 0
            obs = env.reset()
            cnt = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, info = env.step(action)
                score += r[0]
                cnt += 1
            total_score += score
            # modify = # of legal moves in this episode
            # matchcnt = # of matched permutations
            print(f'Episode: {i+1}, Score: {score}, Gate: {cnt}, MatchCnt: {info[0]["MatchCnt"][-1]} Modify: {info[0]["Modify"]}')
            gates = info[0]["GateTrace"]
            for ele in np.transpose(gates):
                print(ele)
            
        print(f'the average score is {total_score / 10}')
        ### Save best model
        if current_best < total_score / 10:
            print("Saving Model")
            current_best = total_score / 10
            save_path = 'models'
            model.save(f"{save_path}/best")

        print("---------------")

        print(current_best)


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env,
        verbose=0,
        learning_rate=0.0003,
        #tensorboard_log=my_config["run_id"]
    )
    train(env, model, my_config)