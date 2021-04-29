#!/usr/bin/env python3

# %%
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

from simulation.epidemic_simulation_array_idx import SimulationA


# %%
class Plot:

    # remove percents and generate them here
    def __init__(self, percents: list, steps: list[int], iter_per_step: int, title: str):
        self.percents     = percents
        self.steps        = steps
        self.iter_per_step= iter_per_step
        self.title        = title

        self.sim_idx      = 0
        self.current_step = 0
        num_steps         = len(percents)
        self.num_per_step = np.zeros(iter_per_step)
        self.ts_avg       = np.zeros(num_steps)
        self.ts_min       = np.zeros(num_steps)
        self.ts_max       = np.zeros(num_steps)
        self.ts_std       = np.zeros(num_steps)

    def add_result(self, num: int):
        self.num_per_step[self.sim_idx] = num
        self.sim_idx += 1
        if self.sim_idx >= self.iter_per_step:
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
        plt.xlabel("Pop. density")
        plt.ylabel(self.title)

        plt.plot(self.percents, self.ts_avg, color="black", marker="o", linestyle="none")
        plt.vlines(self.percents, ymin=self.ts_avg - self.ts_std/2, ymax=self.ts_avg + self.ts_std/2, colors="black")

        plt.bar(x=self.percents, height=self.ts_max-self.ts_min, bottom=self.ts_min, color="lightgrey", width=min(self.steps))

    def save(self, path: Path, filename: str):
        files = Path(path).glob(filename + "*.png")
        num = len(list(files)) + 1
        file_path = path.joinpath(filename+ str(num) + ".png")
        plt.savefig(file_path)


# %%
def main():
    # population density graph config
    # TODO: get from argparse?
    SEED          = 123 # TODO: what should I do with this?
    N             = 128
    MAX_ITER      = 10000
    BREAKPOINTS   = [0.0, 0.01, 0.02, 0.1, 0.6]
    STEPS         = [0.0002, 0.001, 0.005, 0.05]
    LOW_LIMIT     = BREAKPOINTS[0]
    HIGH_LIMIT    = BREAKPOINTS[-1]
    ITER_PER_STEP = 400

    # build range with varying step size
    ranges = []
    br_last = LOW_LIMIT

    for br_i, step in zip(BREAKPOINTS[1:], STEPS):
        _range = np.arange(br_last, br_i, step)
        ranges.append(_range)
        br_last = br_i

    if ranges[-1][-1] != HIGH_LIMIT:
        ranges.append([HIGH_LIMIT])

    percents = np.concatenate(ranges)

    # building string for logging
    num_steps = len(percents)
    number_of_digits = lambda number: len(str(number))
    current_progress = "\r| {:>" + str(number_of_digits(num_steps)) + "}/" +\
        "{} | " +\
        "{:>" + str(number_of_digits(int(100*HIGH_LIMIT)) + 3) + ".2f}% | " +\
        "{:>" + str(number_of_digits(ITER_PER_STEP)) + "}/" +\
        "{:>" + str(number_of_digits(ITER_PER_STEP)) + "} |"

    # run simulations
    death_rate = Plot(percents, steps=STEPS, iter_per_step=ITER_PER_STEP, title="Average death rate")
    num_iter   = Plot(percents, steps=STEPS, iter_per_step=ITER_PER_STEP, title="Average duration")

    for iteration, percent in enumerate(percents):
        M = int(percent * N * N)
        #dead_per_step = np.zeros(ITER_PER_STEP)

        # load simulations from csv
        # TODO: Move this pattern matching to Simulation class
        file_path = Path("/home/janek/code/PG/magisterka/repo/simulations/")
        name_format = "{}N-{}M.csv".format(N, M)
        num_saved = 0
        try:
            with Path(file_path).joinpath(name_format).open() as sim_file:
                for sim_line in sim_file:
                    sim_i = SimulationA(N=N, M=M, csv_line=sim_line)

                    death_rate.add_result(sim_i.ts_dead[sim_i.last_iter] / M if M != 0 else 0)
                    num_iter.add_result(sim_i.last_iter)

                    num_saved += 1
                    if num_saved == ITER_PER_STEP:
                        break
        except FileNotFoundError:
            pass

        # run the remaining simulations
        if ITER_PER_STEP > num_saved:
            sim = SimulationA(N=N, M=M, max_iter=MAX_ITER)
            threads = cpu_count()
            with Pool(threads) as pool:
                seeds = np.full(ITER_PER_STEP - num_saved, SEED)
                print(current_progress.format(iteration +  1, num_steps, percent * 100, num_saved, ITER_PER_STEP), end="")
                for sim_i in pool.imap_unordered(sim.run, seeds):
                    last_iter = sim_i.last_iter
                    #ts_sick   = sim_i.ts_sick
                    ts_dead   = sim_i.ts_dead

                    death_rate.add_result(ts_dead[last_iter] / M if M != 0 else 0)
                    num_iter.add_result(last_iter)

                    sim_i.dump_to_csv(file_path)

                    num_saved += 1
                    print(current_progress.format(iteration +  1, num_steps, percent * 100, num_saved, ITER_PER_STEP), end="")
        print(current_progress.format(iteration +  1, num_steps, percent * 100, num_saved, ITER_PER_STEP))

    path = Path("/home/janek/code/PG/magisterka/repo/test_figures/")
    death_rate.plot()
    death_rate.save(path, "density-death-")

    num_iter.plot()
    num_iter.save(path, "density-iter-")


# %%
if __name__ == "__main__":
    main()