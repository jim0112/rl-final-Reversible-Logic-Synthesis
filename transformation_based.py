import random
n = 3
Y = [*range(1 << n)]
random.shuffle(Y)

# gate的表示法:
# 每個gate均用一個長度是n+1的list表示。list的第一的元素代表要被這個gate施加NOT運算的qbit編號。
# 接下來的n個元素只會是0或1，代表每個qbit是否要當這個gate的control bit。0代表empty，1代表control。
# 如果有qbit會被施加NOT運算且也會當control bit，則該qbit實際上不會用於control。
# ex. [0, 0, 1, 0, 0, 1] 代表qbit編號1、4作control，對qbit編號0施加NOT運算
def output2gates(n, Y):
    gates = []
    for X in range(1 << n):
        if X == Y[X]:
            continue
        new_gates = []
        for i in range(n):
            if X & (1 << i) > 0 and Y[X] & (1 << i) == 0:
                new_gates.append([i, *map(int, reversed(bin(Y[X])[2:].zfill(n)))])
        for i in range(n):
            if X & (1 << i) == 0 and Y[X] & (1 << i) > 0:
                new_gates.append([i, *map(int, reversed(bin(X)[2:].zfill(n)))])
        gates = [*new_gates[::-1], *gates]
        for gate in new_gates:
            ctrl = int(''.join(map(str, gate[:0:-1])), 2) & (((1 << n) - 1) ^ (1 << gate[0]))
            for x in range(X, 1 << n):
                if ctrl & Y[x] == ctrl:
                    Y[x] +=  (2 * int(Y[x] & (1 << gate[0]) == 0) - 1) * (1 << gate[0])
    return gates

def gates2output(n, gates):
    Y = [*range(1 << n)]
    for gate in gates:
        ctrl = int(''.join(map(str, gate[:0:-1])), 2) & (((1 << n) - 1) ^ (1 << gate[0]))
        for i in range(1 << n):
            if ctrl & Y[i] == ctrl:  # apply NOT
                Y[i] += (2 * int(Y[i] & (1 << gate[0]) == 0) - 1) * (1 << gate[0])
    return Y

Y = [5, 3, 2, 4, 8, 7, 6, 1]
gates = output2gates(3, Y)
print(gates)
print(gates2output(3, gates))