#!/usr/bin/env python3

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

    # plot data
    LAST_ITER = 0


    def __init__(self, N, M, L, max_iter):
        self.N        = N     # grid size
        self.M        = M    # population size
        self.L        = L      # lifespan
        self.max_iter = max_iter

        # plot data
        self.TS_SICK  = np.zeros(self.max_iter)


    def run(self, seed):
        # TODO: SEED RANDOM NUMBER GENERATOR!!!

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
            # TODO: Test if removing dead population to speed up later iterations is worth it
            # TODO: Research optimal sorting algorithm (might be different for levy flights)
            #       from what I remember quick sort tends to be slow with pre-sorted data,
            #       so we could improve here
            infected_coords = None
            # on any given coordinates the infected are guaranteed to appear before the healthy
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
                        person["lifespan"] = self.L # not needed?
                        n_sick += 1
            
            # save plot data
            self.TS_SICK[iteration] = n_sick
            iteration += 1
            print(f"I:{iteration}, sick:{round(n_sick / self.M * 100, 2)}%, dead:{round(n_dead / self.M * 100, 2)}%.")

        self.LAST_ITER = iteration


    def show_plot(self):
        plt.plot(np.arange(self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10]/self.M * 100) # percent
        # plt.plot(range(0, self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10])
        plt.show()

    
    def save_plot(self):
        path="/home/janek/code/PG/magisterka/repo/test_figures/sort-"
        files = glob.glob(path + "*.png")
        num = len(files) + 1

        plt.plot(np.arange(self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10]/self.M * 100) # percent
        # plt.plot(range(0, self.LAST_ITER + 10), self.TS_SICK[0:self.LAST_ITER + 10])
        plt.savefig(path + str(num) + ".png")


if __name__ == "__main__":
    simulation = Simulation(N=128, M=4000, L=20, max_iter=10000)
    simulation.run(123)
    simulation.show_plot()
