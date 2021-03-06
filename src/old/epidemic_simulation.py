#!/usr/bin/env python3
# Sorting verion of epidemics simulation

import numpy as np
import matplotlib.pyplot as plt

import glob


class Simulation:

    # possible movements
    X_STEP = np.array([-1, 0, 1, 0])
    Y_STEP = np.array([0, -1, 0, 1])

    # "infect" column possible states (order inversed for performance reasons)
    DEAD     = 0
    INFECTED = 1
    HEALTHY  = 2


    def __init__(self, N, M, L, max_iter, logging=False):
        self.N        = N      # grid size
        self.M        = M      # population size
        self.L        = L      # lifespan
        self.max_iter = max_iter
        self.logging = logging

        # plot data
        self.TS_SICK  = np.zeros(self.max_iter)
        self.TS_DEAD = np.zeros(self.max_iter)
        self.LAST_ITER = 0


    def run(self, seed):

        if self.N <= 0 or self.M <= 0:
            return {
                "LAST_ITER": self.LAST_ITER,
                "TS_SICK": self.TS_SICK,
                "TS_DEAD": self.TS_DEAD
            }

        # population information
        dtype = [
            ("x", int),
            ("y", int),
            ("infect", int),
            ("lifespan", int)
        ]

        population = np.array([
            (
                np.random.randint(0, self.N), # x
                np.random.randint(0, self.N), # y
                self.HEALTHY,                 # infect status
                self.L                        # lifespan
            )
            for _
            in np.arange(self.M)
        ], dtype=dtype)

        # infecting a random person
        jj = np.random.randint(0, self.M-1)
        population[jj]["infect"] = self.INFECTED

        # plot data
        n_sick, n_dead = 1, 0
        iteration = 0

        # main loop
        while (n_sick > 0) and (iteration < self.max_iter):
            # move all actors
            for person in population:
                if person["infect"] != self.DEAD:
                    ii = np.random.randint(4)
                    person["x"] += self.X_STEP[ii]
                    person["y"] += self.Y_STEP[ii]
                    person["x"] = min(self.N, max(person["x"], 1))
                    person["y"] = min(self.N, max(person["y"], 1))

            # deal with infected
            infected_coords = None
            population.sort(order=["x", "y", "infect"])
            for person in population:
                if person["infect"] == self.DEAD:
                    pass

                elif person["infect"] == self.INFECTED:
                    infected_coords = (person["x"], person["y"])
                    person["lifespan"] -= 1

                    if person["lifespan"] <= 0:
                        person["infect"] = self.DEAD
                        n_sick -= 1
                        n_dead += 1

                elif person["infect"] == self.HEALTHY:
                    if (person["x"], person["y"]) == infected_coords:
                        person["infect"] = self.INFECTED
                        person["lifespan"] = self.L
                        n_sick += 1

            # save plot data
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
        plt.plot(np.arange(self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10]/self.M * 100)
        plt.show()


    def save_plot(self, path):
        files = glob.glob(path + "*.png")
        num = len(files) + 1

        plt.plot(np.arange(self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10]/self.M * 100)
        plt.savefig(path + str(num) + ".png")


if __name__ == "__main__":
    simulation = Simulation(N=128, M=128**2, L=20, max_iter=10000, logging=True)
    simulation.run(123)
    simulation.show_plot()
