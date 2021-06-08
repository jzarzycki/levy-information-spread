#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N        = 128
M        = 4000
L        = 20
max_iter = 10000

x_step     = np.array([-1, 0, 1, 0])
y_step     = np.array([0, -1, 0, 1])
x, y       = np.zeros(M), np.zeros(M)
infect     = np.zeros(M)
lifespan   = np.zeros(M)
ts_sick    = np.zeros(max_iter)

for j in range(M):
    x[j] = np.random.randint(0, N)
    y[j] = np.random.randint(0, N)
    lifespan[j] = L

jj = np.random.randint(0, M-1)
infect[jj] = 1
n_sick, n_dead, iterate = 1, 0, 0

while (n_sick > 0) and (iterate < max_iter):
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
                infect[j] = 2
                n_sick -= 1
                n_dead += 1

            for k in range(0, M):
                if infect[k] == 0 and k != j:
                    if x[j] == x[k] and y[j] == y[k]:
                        infect[k] = 1
                        lifespan[k] = L
                        n_sick += 1

    ts_sick[iterate] = n_sick
    iterate += 1
    print("iteration {0}, sick {1}, dead {2}".format(iterate, n_sick, n_dead))

plt.plot(range(0, iterate + 10), ts_sick[0:iterate + 10])
plt.show()

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
