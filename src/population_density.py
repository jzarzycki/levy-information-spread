#!/usr/bin/env python3

# TODO: add reading simulation data from csv (mock class?)

# %%
import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

import simulation.epidemic_simulation_array_idx as ep_simulation

# %%
# Population density graph config
LOW_LIMIT     = 0
BREAKPOINTS   = [0.1,  0.2,   0.23, 0.27,   0.3,  0.4,   0.6]
STEP          = [0.05, 0.025, 0.01, 0.0025, 0.01, 0.025, 0.05]
#BREAKPOINTS   = [0.2, 0.3, 0.4]
#STEP          = [0.1, 0.05, 0.1]
HIGH_LIMIT    = max(BREAKPOINTS)
ITER_PER_STEP = 20

N        = 128
L        = 20
MAX_ITER = 10000
SEED = 123 # what should I do with this?

# %%
class Plot:

    def __init__(self, percents: list, title: str):
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

    def add_result(self, num: int):
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

    def plot(self, widths: list=None):
        plt.figure()
        fig, ax = plt.subplots()
        ax.set_xlabel("Pop. density")
        ax.set_ylabel(self.title)

        ax.plot(self.percents, self.ts_avg, color="black", marker="o", linestyle="none")
        ax.vlines(self.percents, ymin=self.ts_avg - self.ts_std/2, ymax=self.ts_avg + self.ts_std/2, colors="black")

        args = {
            "x": self.percents,
            "height": self.ts_max - self.ts_min,
            "bottom": self.ts_min,
            "color": "lightgrey"
        }
        if widths is not None:
            widths = np.array(widths)
            left_width = -widths[:-1] / 2
            right_width = widths[1:] / 2

            ax.bar(**args, width=left_width, align="edge")
            ax.bar(**args, width=right_width, align="edge")
        else:
            ax.bar(**args, width=min(STEP))

    def save(self, path: str):
        files = glob.glob(path + "*.png")
        num = len(files) + 1
        plt.savefig(path + str(num) + ".png")


# %%
if __name__ == "__main__":
    # build range with varying step size
    ranges = []
    widths = []
    br_last = LOW_LIMIT
    for br_i, step in zip(BREAKPOINTS, STEP):
        _range = np.arange(br_last, br_i, step)
        ranges.append(_range)
        widths += [step] * len(_range)
        br_last = br_i
    # tutaj wygeneruj fix do plt.bar z ranges
    if ranges[-1][-1] != BREAKPOINTS[-1]:
        widths.append(BREAKPOINTS[-1] - ranges[-1][-1])
        ranges.append([BREAKPOINTS[-1]])
    percents = np.concatenate(ranges)
    widths = [widths[0], *widths] # move this to the class

    num_steps = len(percents)
    number_of_digits = lambda number: len(str(number))
    current_progress = "\r| {:>" + str(number_of_digits(num_steps)) + "}/" +\
        "{} | " +\
        "{:>" + str(number_of_digits(int(100*HIGH_LIMIT)) + 3) + ".2f}% | " +\
        "{:>" + str(number_of_digits(ITER_PER_STEP)) + "}/" +\
        "{:>" + str(number_of_digits(ITER_PER_STEP)) + "} |"

    death_rate = Plot(percents, title="Average death rate")
    num_iter   = Plot(percents, title="Average duration")

    for iteration, percent in enumerate(percents):
        M = int(percent * N * N)

        sim = ep_simulation.Simulation(N=N, M=M, L=L, max_iter=MAX_ITER)
        dead_per_step = np.zeros(ITER_PER_STEP)

        threads = cpu_count()
        with Pool(threads) as pool:
            seeds = np.full(ITER_PER_STEP, SEED)
            num = 0
            print(current_progress.format(iteration+1, num_steps, percent * 100, num, ITER_PER_STEP), end="")
            for result in pool.imap_unordered(sim.run, seeds):
                last_iter = result["LAST_ITER"]
                #ts_sick   = result["TS_SICK"]
                ts_dead   = result["TS_DEAD"]

                death_rate.add_result(ts_dead[last_iter] / M if M != 0 else 0)
                num_iter.add_result(last_iter)

                num += 1
                print(current_progress.format(iteration+1, num_steps, percent * 100, num, ITER_PER_STEP), end="")
            print()

    path = "/home/janek/code/PG/magisterka/repo/test_figures/density-"
    death_rate.plot(widths)
    death_rate.save(path + "death-")

    num_iter.plot(widths)
    num_iter.save(path + "iter-")
