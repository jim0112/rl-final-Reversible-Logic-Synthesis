import numpy as np

def make_base(n):
    res = []
    for i in range(1 << n):
        bits = bin(i)[2:]
        while len(bits) < n:
            bits = '0' + bits
        res.append([int(b) for b in bits])
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

if __name__ == '__main__':
    res = make_base(3)
    print(res)
    print(len(res))
    actions = make_actions(3)
    print(actions)
    print(len(actions))
