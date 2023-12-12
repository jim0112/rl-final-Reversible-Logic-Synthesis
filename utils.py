import numpy as np

def cal_factorial(n):
    n = 1 << n
    ans = 1
    for i in range(1, n+1):
        ans *= i
    return ans

def make_base(n):
    res = []
    for i in range(1 << n):
        bits = bin(i)[2:]
        while len(bits) < n:
            bits = '0' + bits
        res.append([int(b) for b in bits])
    return res

def make_test(n, nums, seed):
    import random
    random_seed = seed
    random.seed(random_seed)
    base = make_base(n)
    shuffle = base.copy()
    res = []
    for _ in range(nums):
        random.shuffle(shuffle)
        res.append(shuffle[:])
    return res

def isvalid(action):
    if 2 not in action:
        return False
    elif action.count(2) > 1:
        return False
    return True

def dfs(i, n, action, actions):
    if i == n:
        if isvalid(action):
            actions.append(action)
        return
    for a in range(3):
        dfs(i+1, n, action + [a], actions) 

def make_actions(n):
    # 0: no, 1: control, 2: not
    # cannot have more than 1 not with rest no
    # must have at least one not
    actions = []
    dfs(0, n, [], actions)
    return actions

def string2index(n):
    a = {}
    for i in range(1 << n):
        bits = bin(i)[2:]
        while len(bits) < n:
            bits = '0' + bits
        a[bits] = i
    return a

# compute mask for every base, so that it can be "and" when in the env
def base_mask(base, actiondict):
    mask_action_dict = np.ones((len(base),len(actiondict)))
    # ignoring the condition when last state (111) is matched, 
    # cause it would definitely mask all the action
    for n in range(len(base)-1):
        state = base[n]
        for act in range(len(actiondict)):
            new_state = []
            control = []
            action = actiondict[act]
            for i, a in enumerate(action):
                if a == 1:
                    control.append(i)

                flip = 1
                for c in control:
                    flip *= state[c]
                tmp = []
                for i, bit in enumerate(state):
                    if flip and action[i] == 2:
                        tmp.append(0 if bit == 1 else 1)
                    else:
                        tmp.append(bit)
            new_state.append(tmp)
            if state != tmp:
                mask_action_dict[n][act] = 0
    return mask_action_dict

if __name__ == '__main__':
    n = 4
    # res = make_base(n)
    # print(res)
    # print(len(res))
    # actions = make_actions(n)
    # print(actions)
    # print(len(actions))
    # a = string2index(n)
    # print(a)
    test = make_test(n, 10, 87)
    
