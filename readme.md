# Reversible Logic synthesis
> 組別 ： 16

## prequisite
```
conda create -n rls_env python=3.10
conda activate rls_env
pip install -r requirements.txt
```

## Project structure
```
/Project root
├── envs/ - containing env
    ├── _init_.py
    ├── RLS_env_eval.py - our enviornments
├── models/ - storing our models
├── train.py - training models
├── env_test.py - compare a models with simple and strong baseline
├── transformation_based.py - generating simple and strong baseline
├── utils.py - containing many useful function
├── requirements.txt - packages need to be installed
├── README.md - This file, providing an overview and guidance for the project.
```

## how to reproduce our results
### 5.1 RL algorithms
- change my_config in train.py
```python
my_config = {
    # select different algorithms: PPO, A2C, DQN, MaskablePPO
    "algorithm": MaskablePPO,
    # select different n
    "n":3,
}
```
- change self.n in RLS_env_eval.py
- change my_config in env_test.py
```python
my_config = {
    # use same algorithm and n in train.py
    "algorithm": MaskablePPO,
    "n" : 3,
}
```

- run command
```bash
python3 train.py
python3 env_test.py
```

### 5.2 Policy Network Architecture
* One can simply change the model layers by the following:
```python
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
```
* After that, put the parameter in MaskablePPO:
```python
model = MaskablePPO(
        my_config["policy_network"], 
        env,
        verbose=0,
        learning_rate=0.0003,
        policy_kwargs=policy_kwargs,
        # tensorboard_log=my_config["run_id"]
    )
```
* run command
```bash
python3 train.py
python3 env_test.py
```

### 5.3 Quantum Bit Size
(this have been included in 5.1 experiments)
- change my_config in train.py
```python
my_config = {
    "algorithm": MaskablePPO,
    # select different n
    "n":3,
}
```
- change self.n in RLS_env_eval.py
- change my_config in env_test.py
```python
my_config = {
    "algorithm": MaskablePPO,
    "n" : 3,
}
```

- run command
```bash
python3 train.py
python3 env_test.py
```
### 5.4 Previous States Amount
For different number of bits:
- change self.n in RLS_env_eval.py
- change the n value in train function in train_mask.py
- change the n value in env_test_mask.py

For differnt previous state amount:
- change self.prev_cnt in RLS_env_eval.py


And then run command:
```bash
python3 train.py
python3 env_test.py
```
