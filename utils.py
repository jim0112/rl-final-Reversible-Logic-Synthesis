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

def string2index(n):
    a = {}
    for i in range(1 << n):
        bits = bin(i)[2:]
        while len(bits) < n:
            bits = '0' + bits
        a[bits] = i
    return a
if __name__ == '__main__':
    n = 4
    res = make_base(n)
    print(res)
    print(len(res))
    actions = make_actions(n)
    print(actions)
    print(len(actions))
    a = string2index(n)
    print(a)
