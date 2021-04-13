#!/usr/bin/env python3

import glob

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from index import PositionIndex
else:
    from simulation.index import PositionIndex


class Simulation:

    HEALTHY  = 0
    INFECTED = 1
    DEAD     = 2

    X_STEP   = np.array([-1, 0, 1, 0], dtype=int)
    Y_STEP   = np.array([0, -1, 0, 1], dtype=int)


    def __init__(self, N, M, L, max_iter, logging=False):
        self.N        = N
        self.M        = M
        self.L        = L
        self.MAX_ITER = max_iter
        self.logging  = logging

        # plot data
        self.TS_SICK   = np.zeros(max_iter, dtype=int)
        self.TS_DEAD   = np.zeros(max_iter, dtype=int)
        self.LAST_ITER = 0


    def run(self, seed):
        # TODO: use/delete the seed param

        if self.N <= 0 or self.M <= 0:
            return {
                    "LAST_ITER": self.LAST_ITER,
                    "TS_SICK": self.TS_SICK,
                    "TS_DEAD": self.TS_DEAD
            }

        x, y     = np.zeros(self.M, dtype=int), np.zeros(self.M, dtype=int)
        infect   = np.zeros(self.M, dtype=int)
        lifespan = np.zeros(self.M, dtype=int)

        # my optimizations
        # TODO: use infcted_list - after implementing SIRE
        # TODO: add exposed_list? - probably not needed - can just keep in infected list and check in code for type
        infected_list = -np.ones(self.M, dtype=int) # fill with minus ones
        index = PositionIndex(grid_x=self.N, grid_y=self.N)


        for j in range(self.M):
            x[j] = np.random.randint(0, self.N)
            y[j] = np.random.randint(0, self.N)
            lifespan[j] = self.L
            index.append_index((x[j], y[j]), j)

        jj = np.random.randint(0, self.M-1)
        infect[jj] = 1
        n_sick, n_dead, iteration = 1, 0, 0
        infected_list[0] = jj

        while (n_sick > 0) and (iteration < self.MAX_ITER):
            # TODO: UNUSED FIX???
            new_infect = infect.copy() # TODO: optimize this bug fix

            for j in range(0, self.M): # TODO: iterate only over infected and exposed (after implementing SIRE)
            
                if infect[j] != self.DEAD:
                    ii = np.random.randint(4)

                    new_x = x[j] + self.X_STEP[ii]
                    new_y = y[j] + self.Y_STEP[ii]
                    new_x = min(self.N - 1, max(new_x, 1))
                    new_y = min(self.N - 1, max(new_y, 1))

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
                    for k in same_position:
                        if infect[k] == 0 and k != j:
                            infect[k] = 1
                            lifespan[k] = self.L
                            n_sick += 1

            self.TS_SICK[iteration] = n_sick
            self.TS_DEAD[iteration] = n_dead
            iteration += 1
            if self.logging:
                print(f"I:{iteration}, sick:{round(n_sick / self.M * 100, 2)}%, dead:{round(n_dead / self.M * 100, 2)}%.")

        self.LAST_ITER = iteration - 1
        return {
                "LAST_ITER": self.LAST_ITER,
                "TS_SICK": self.TS_SICK,
                "TS_DEAD": self.TS_DEAD
        }


    def show_plot(self):
        plt.plot(np.arange(self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10]/self.M * 100) # percent
        plt.show()


    def save_plot(self, path):
        files = glob.glob(path + "*.png")
        num = len(files) + 1

        plt.plot(np.arange(self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10]/self.M * 100) # percent
        # plt.plot(range(0, self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10])
        plt.savefig(path + str(num) + ".png")


if __name__ == "__main__":
    simulation = Simulation(N=128, M=128**2, L=20, max_iter=10000, logging=True)
    simulation.run(123)
    simulation.show_plot()