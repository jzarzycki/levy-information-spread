#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N        = 128
M        = N**2
#M        = 4000
L        = 20
max_iter = 10000

HEALTHY  = 0
INFECTED = 1
DEAD     = 2

x_step   = np.array([-1, 0, 1, 0])
y_step   = np.array([0, -1, 0, 1])

x, y     = np.zeros(M), np.zeros(M)
infect   = np.zeros(M)
lifespan = np.zeros(M)

# my optimizations
# TODO: add exposed_list
infected_list = -np.ones(M) # fill with minus ones

class PositionIndex:

    def __init__(self, grid_x, grid_y=1):
        self.__grid_x = grid_x
        self.__grid_y = grid_y
        self.__index = {}
        # self.__index = [[]] * (grid_x * grid_y + 1),
        #self.__index = np.array(
        #    [[]] * (grid_x * grid_y + 1),
        #    dtype=object
        #)
        
    def get_index(self, pos):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")
        #current_pos = int((self.__grid_x-1) * pos[1] + pos[0])
        #return self.__index[current_pos]
        return self.__index[pos]
    
    def append_index(self, pos, actor_idx):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")
        
        try:
            idx = self.__index[pos]
        except KeyError:
            idx = []

        idx.append(actor_idx)
        self.__index[pos] = idx

        #current_pos = int((self.__grid_x-1) * pos[1] + pos[0])
        #try:
        #    idx = self.__index[current_pos]
        #    #idx = np.append(idx, actor_idx)
        #    idx.append(actor_idx)
        #    print(idx, actor_idx)
        #    self.__index[current_pos] = idx
        #    print(self.__index[current_pos])
        #    print()
        #except IndexError as e:
        #    print(pos, self.__grid_x, current_pos)
        #    print(len(self.__index))
        #    raise e
        
    def remove_index(self, pos, actor_idx):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")

        try:
            idx = self.__index[pos]
            self.__index[pos] = [
                i for i
                in idx
                if i != actor_idx
            ]
        except KeyError:
            return
        #current_pos = int(pos[1] * (self.__grid_x-1) + pos[0])
        #idx = self.__index[current_pos]
        #self.__index[current_pos] = idx[idx != actor_idx]


index = PositionIndex(grid_x=N, grid_y=N)

ts_sick  = np.zeros(max_iter)

for j in range(M):
    x[j] = np.random.randint(0, N)
    y[j] = np.random.randint(0, N)
    lifespan[j] = L
    index.append_index((x[j], y[j]), j)

jj = np.random.randint(0, M-1)
infect[jj] = 1
n_sick, n_dead, iterate = 1, 0, 0
infected_list[0] = jj

while (n_sick > 0) and (iterate < max_iter):
    new_infect = infect.copy()

    # move actors
    # update position index
    # handle infected

    for j in range(0, M): # TODO: iterate to M - dead (later just iterate over infected and exposed)

        if infect[j] != DEAD:
            ii = np.random.choice([0, 1, 2, 3])

            new_x = x[j] + x_step[ii]
            new_y = y[j] + y_step[ii]
            new_x = min(N, max(new_x, 1))
            new_y = min(N, max(new_y, 1))

            index.remove_index((x[j], y[j]), j)
            index.append_index((new_x, new_y), j)
            
            x[j] = new_x
            y[j] = new_y

        if infect[j] == 1:

            lifespan[j] -= 1

            if lifespan[j] <= 0:
                infect[j] = 2
                n_sick -= 1
                n_dead += 1
                index.remove_index((x[j], y[j]), j)

            same_position = index.get_index((x[j], y[j]))
            #print(same_position)
            for k in same_position:
                if infect[k] == 0 and k != j:
                    infect[k] = 1
                    lifespan[k] = L
                    n_sick += 1

    ts_sick[iterate] = n_sick
    iterate += 1
    print(f"I:{iterate}, sick:{round(n_sick / M * 100, 2)}%, dead:{round(n_dead / M * 100, 2)}%.")

MODE = 'show'
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
