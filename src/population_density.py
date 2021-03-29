#!/usr/bin/env python3

# TODO: add reading simulation data from csv (mock class?)

# %%

import simulation.epidemic_simulation as ep_simulation

import numpy as np
import matplotlib.pyplot as plt

import glob
from multiprocessing import Pool, cpu_count

# %%

# Population density graph config
LOW_LIMIT     = 0.0
HIGH_LIMIT    = 0.6
STEP          = 0.1
ITER_PER_STEP = 8

N        = 128
L        = 20
MAX_ITER = 10000

# %%
class Plot:
    
    def __init__(self, percents, title):
        self.sim_idx = 0
        self.num_per_step = np.zeros(ITER_PER_STEP)
        self.percents = percents
        self.title = title

        self.current_step = 0
        num_steps = len(percents)
        self.ts_avg = np.zeros(num_steps)
        self.ts_min = np.zeros(num_steps)
        self.ts_max = np.zeros(num_steps)
        self.ts_std = np.zeros(num_steps)

    def add_result(self, num):
        self.num_per_step[self.sim_idx] = num
        self.sim_idx += 1
        if self.sim_idx >= ITER_PER_STEP:
            self._next_step()

    def _next_step(self):
        self.ts_avg[self.current_step] = np.average(self.num_per_step)
        self.ts_min[self.current_step] = np.min(self.num_per_step)
        self.ts_max[self.current_step] = np.max(self.num_per_step)
        self.ts_std[self.current_step] = np.std(self.num_per_step)

        self.sim_idx = 0
        self.current_step += 1

    def plot(self):
        plt.figure()
        fig, ax = plt.subplots()
        ax.set_xlabel("Pop. density")
        ax.set_ylabel(self.title)

        ax.plot(self.percents, self.ts_avg, color="black", marker="o", linestyle="none")
        ax.vlines(self.percents, ymin=self.ts_avg - self.ts_std/2, ymax=self.ts_avg + self.ts_std/2, colors="black")
        ax.bar(self.percents, height=self.ts_max - self.ts_min, bottom=self.ts_min, width=STEP, color="lightgrey")

    def save(self, path):
        files = glob.glob(path + "*.png")
        num = len(files) + 1
        plt.savefig(path + str(num) + ".png")


# %%
if __name__ == "__main__":
    percents = np.arange(LOW_LIMIT, HIGH_LIMIT + STEP, STEP)
    num_steps = len(percents)

    death_rate = Plot(percents, title="Average death rate")
    num_iter   = Plot(percents, title="Average duration")

    for iteration, percent in enumerate(percents):
        M = int(percent * N * N)
        print("pop. density =", str(round(percent * 100, 2)) + "% / " + str(round(HIGH_LIMIT * 100, 2)) + "%", "(M=" + str(M) + ")")

        sim = ep_simulation.Simulation(N=N, M=M, L=L, max_iter=MAX_ITER)
        dead_per_step = np.zeros(ITER_PER_STEP)

        threads = cpu_count()
        with Pool(threads) as pool:
            seeds = [123] * ITER_PER_STEP
            results = pool.map(sim.run, seeds)
            for i, result in enumerate(results):
                last_iter = result["LAST_ITER"]
                #ts_sick   = result["TS_SICK"]
                ts_dead   = result["TS_DEAD"]

                death_rate.add_result(ts_dead[last_iter] / M if M != 0 else 0)
                num_iter.add_result(last_iter)

    path = "/home/janek/code/PG/magisterka/repo/test_figures/density-"
    death_rate.plot()
    death_rate.save(path + "death-")

    num_iter.plot()
    num_iter.save(path + "iter-")
