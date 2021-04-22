#!/usr/bin/env python3

import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # a hack, that fixes imports, when run as a script
    import sys
    sys.path.append(r"/home/janek/code/PG/magisterka/repo/src")
from simulation.index import PositionIndex
from distribution.levy import Levy


class Simulation:

    HEALTHY  = 0
    INFECTED = 1
    DEAD     = 2


    def __init__(self, N: int=None, M: int=None, L: int=None, max_random_step: int=10,
                    max_iter: int=10000, logging: bool=False, csv_line: str=None):
        self.N = N
        self.M = M
        self.L = L

        #self.max_random_step = max_random_step # TODO: wywalić to?
        self.levy = Levy(max_random_step)

        if csv_line is None:
            self.max_iter = max_iter
            self.logging  = logging

            self.ts_sick   = np.zeros(max_iter, dtype=int)
            self.ts_dead   = np.zeros(max_iter, dtype=int)
            self.last_iter = 0
        else:
            str_sick, str_dead = csv_line[:-1].split(";")
            self.ts_sick = np.array([int(value) for value in str_sick.split(",")])
            self.ts_dead = np.array([int(value) for value in str_dead.split(",")])
            self.last_iter = len(self.ts_sick) - 1


    def run(self, seed: int):
        # TODO: use/delete the seed param

        if self.N <= 0 or self.M <= 0:
            return self

        x, y     = np.zeros(self.M, dtype=int), np.zeros(self.M, dtype=int)
        infect   = np.zeros(self.M, dtype=int)
        lifespan = np.zeros(self.M, dtype=int)

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

        while (n_sick > 0) and (iteration < self.max_iter):
            # create an empty set to keep info on newly infected actors,
            # so we don't mutate the array we are looping over
            # TODO: replace with another infected index?
            new_infect = set() # TODO: try replacing with np.array?

            for j in range(0, self.M): # TODO: iterate only over infected and exposed (after implementing SIRE)

                if infect[j] != self.DEAD:
                    new_x = x[j] + self.levy.random_step()
                    new_y = y[j] + self.levy.random_step()

                    new_x %= self.N
                    new_y %= self.N

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

                    same_position = index[(x[j], y[j])]
                    for k in same_position:
                        if infect[k] == 0 and k != j:
                            # keep track of newly infected in a set
                            new_infect.add(k)

            # update the newly infected status
            for idx in new_infect:
                infect[idx] = 1
                lifespan[idx] = self.L
                n_sick += 1

            self.ts_sick[iteration] = n_sick
            self.ts_dead[iteration] = n_dead
            iteration += 1
            if self.logging:
                print(f"I:{iteration}, sick:{round(n_sick / self.M * 100, 2)}%, dead:{round(n_dead / self.M * 100, 2)}%.")

        self.last_iter = iteration - 1
        return self


    def plot(self):
        fig, ax = plt.subplots()
        ax.title.set_text("Epidemic simulation")
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("% of infected population")
        plt.plot(np.arange(self.last_iter), self.ts_sick[0:self.last_iter]/self.M * 100) # percent


    def show(self):
        plt.show()


    def save(self, path: str):
        files = glob.glob(path + "*.png")
        num = len(files) + 1
        plt.savefig(path + str(num) + ".png")


    def dump_to_csv(self, path: str):
        with open("{}/{}N-{}M-{}L.csv".format(path, self.N, self.M, self.L), "a") as f:

            for idx in np.arange(self.last_iter):
                f.write(str(self.ts_sick[idx]))
                if idx != self.last_iter - 1:
                    f.write(",")
                else:
                    f.write(";")

            for idx in np.arange(self.last_iter):
                f.write(str(self.ts_dead[idx]))
                if idx != self.last_iter - 1:
                    f.write(",")
                else:
                    f.write("\n")


if __name__ == "__main__":
    if "--load" in sys.argv[1:]:
        N, M, L = 128, 16384, 20
        file = "/home/janek/code/PG/magisterka/repo/simulations/_{}-{}-{}.csv".format(N, M, L)

        with open(file) as f:
            for line in f:
                simulation = Simulation(N=N, M=M, L=L, csv_line=line)
                simulation.plot()
                simulation.show()
    else:
        simulation = Simulation(N=128, M=128**2, L=20, max_iter=10000, logging=True)
        simulation.run(123)
        simulation.dump_to_csv("/home/janek/code/PG/magisterka/repo/simulations/")
        simulation.plot()
        simulation.show()
