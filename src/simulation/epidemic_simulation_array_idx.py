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


class SimulationA:

    SUSCEPTIBLE = 0
    INFECTED    = 1
    RECOVERED   = 2


    def __init__(self, N: int=None, M: int=None, max_random_step: int=10,
                    max_iter: int=10000, logging: bool=False, csv_line: str=None):
        self.N = N
        self.M = M

        self.random_walk = Levy(max_random_step)

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


    def handle_two_infected_meet(self) -> tuple:
        return self.RECOVERED, self.RECOVERED


    def run(self, seed: int):
        # TODO: use/delete the seed param

        if self.N <= 0 or self.M <= 0:
            return self

        x, y   = np.zeros(self.M, dtype=int), np.zeros(self.M, dtype=int)
        infect = np.zeros(self.M, dtype=int)

        position_index = PositionIndex(grid_x=self.N, grid_y=self.N)
        infected_index = set()
        new_infected   = set()
        new_recovered  = set()

        for j in range(self.M):
            x[j] = np.random.randint(0, self.N)
            y[j] = np.random.randint(0, self.N)
            position_index.append_index((x[j], y[j]), j)

        jj = np.random.randint(0, self.M-1)
        infect[jj] = 1
        n_heal, n_sick, n_dead, iteration = self.M - 1, 1, 0, 0
        infected_index.add(jj)

        while (n_sick != 0) and (n_heal + n_sick != 1) and (iteration < self.max_iter):

            for j in infected_index:

                if infect[j] == self.INFECTED:
                    # move
                    position_index.remove_index((x[j], y[j]), j)
                    new_x = x[j] + self.random_walk.random_step()
                    new_y = y[j] + self.random_walk.random_step()
                    x[j] = new_x % self.N
                    y[j] = new_y % self.N
                    position_index.append_index((new_x, new_y), j)

                    # save which walkers changed status
                    same_position = position_index[(x[j], y[j])]
                    for k in same_position:
                        if infect[k] == self.SUSCEPTIBLE and j != k:
                            new_infected.add(k)
                        elif infect[k] == self.RECOVERED and j != k:
                            new_recovered.add(j)
                        elif infect[k] == self.INFECTED and j != k:
                            status1, status2 = self.handle_two_infected_meet()

                            if status1 == self.RECOVERED:
                                new_recovered.add(j)
                            if status2 == self.RECOVERED:
                                new_recovered.add(k)

            # update walker statuses
            for idx in new_infected:
                infect[idx] = self.INFECTED
                n_sick += 1
                n_heal -= 1
                infected_index.add(idx)
            new_infected.clear()

            for idx in new_recovered:
                infect[idx] = self.RECOVERED
                n_sick -= 1
                n_dead += 1
                position_index.remove_index((x[idx], y[idx]), idx)
            new_recovered.clear()

            # plot data
            self.ts_sick[iteration] = n_sick
            self.ts_dead[iteration] = n_dead
            iteration += 1

            if self.logging:
                print(f"I:{iteration}, healthy: {self.M - n_sick - n_dead}, sick:{n_sick}, dead:{n_dead}")
                #print(f"I:{iteration}, sick:{round(n_sick / self.M * 100, 2)}%, dead:{round(n_dead / self.M * 100, 2)}%.")

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
        with open("{}/{}N-{}M.csv".format(path, self.N, self.M), "a") as f:

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


class SimulationB(SimulationA):

    def handle_two_infected_meet(self):
        return self.INFECTED, self.RECOVERED


if __name__ == "__main__":
    directory = "/home/janek/code/PG/magisterka/repo/simulations/"

    N = 128
    M = N ** 2

    if "--load" in sys.argv[1:]:
        file = "_{}-{}.csv".format(N, M)

        with open(directory + file) as f:
            for line in f:
                simulation = SimulationA(N=N, M=M, csv_line=line)
                simulation.plot()
                simulation.show()
    else:

        simulation = SimulationA(N=N, M=M, max_iter=10000, logging=True)
        simulation.run(123)
        simulation.dump_to_csv(directory)

        simulation.plot()
        simulation.show()
