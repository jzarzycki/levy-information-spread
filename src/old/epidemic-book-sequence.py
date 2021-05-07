#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N        = 128
M        = 4000
L        = 20 + 1
max_iter = 10000

x_step   = np.array([-1, 0, 1, 0])
y_step   = np.array([0, -1, 0, 1])
x, y     = np.zeros(M), np.zeros(M)
new_infect   = np.zeros(M)
lifespan = np.zeros(M)
ts_sick  = np.zeros(max_iter)

for j in range(M):
    x[j] = np.random.randint(0, N)
    y[j] = np.random.randint(0, N)
    lifespan[j] = L

jj = np.random.randint(0, M-1)
new_infect[jj] = 1
n_sick, n_dead, iterate = 1, 0, 0

while (n_sick > 0) and (iterate < max_iter):
    infect = new_infect.copy()

    for j in range(0, M):

        if infect[j] < 2:
            ii = np.random.choice([0, 1, 2, 3])
            x[j] += x_step[ii]
            y[j] += y_step[ii]
            x[j] = min(N, max(x[j], 1))
            y[j] = min(N, max(y[j], 1))

        if infect[j] == 1:

            lifespan[j] -= 1

            if lifespan[j] <= 0:
                new_infect[j] = 2
                n_sick -= 1
                n_dead += 1

    for j in range(0, M):
        if infect[j] == 1:
            for k in range(0, M):
                if new_infect[k] == 0 and k != j:
                    if x[j] == x[k] and y[j] == y[k]:
                        new_infect[k] = 1
                        lifespan[k] = L
                        n_sick += 1

    ts_sick[iterate] = n_sick
    iterate += 1
    print(f"I:{iterate}, sick:{round(n_sick / M * 100, 2)}%, dead:{round(n_dead / M * 100, 2)}%.")

#MODE = 'show'
MODE = 'save'
if MODE == 'show':
    plt.plot(np.arange(iterate + 10), ts_sick[0:iterate + 10]/M * 100) # percent
    # plt.plot(range(0, iterate + 10), ts_sick[0:iterate + 10])
    plt.show()
elif MODE == 'save':
    import glob
    path="/home/janek/code/PG/magisterka/repo/test_figures/orig-"
    files = glob.glob(path + "*.png")
    num = len(files) + 1

    plt.plot(np.arange(iterate + 10), ts_sick[0:iterate + 10]/M * 100) # percent
    # plt.plot(range(0, iterate + 10), ts_sick[0:iterate + 10])
    plt.savefig(path + str(num) + ".png")
