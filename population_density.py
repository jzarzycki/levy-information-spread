#!/usr/bin/env python3

"""
Module for running simulations for multiple values of population density and plotting their results.
"""

# %%
from pathlib import Path
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from src.info_spread import Simulation, SimulationA, SimulationB


# %%
class Plot:

    def __init__(self, percents: list, steps: list, iter_per_step: int, title: str):
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

    @staticmethod
    def make_plot_steps(breakpoints, steps):
        """make an np.array out of breakpoints and steps arrays"""

        low_limit = breakpoints[0]
        high_limit = breakpoints[-1]
        ranges = []
        br_last = low_limit

        for br_i, step in zip(breakpoints[1:], steps):
            _range = np.arange(br_last, br_i, step)
            ranges.append(_range)
            br_last = br_i

        if ranges[-1][-1] != high_limit:
            ranges.append([high_limit])

        percents = np.concatenate(ranges)
        return percents


    def add_result(self, num: int):
        self.num_per_step[self.sim_idx] = num
        self.sim_idx += 1
        if self.sim_idx >= self.iter_per_step:
            self.__next_step()

    def __next_step(self):
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

    def show(self):
        plt.show()


def parse_arguments():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--csv-dir", default=None, help="path to directory, where simulation results are to be kept")
    parser.add_argument("--plot-dir", default=None, help="path to directory, where plots should be saved")

    parser.add_argument("--breakpoints", type=float, default=[], nargs="+", help="points of the plot, where step sizes change")
    parser.add_argument("--step-sizes", type=float, default=[], nargs="+", help="size of the steps between consecutive breakpoints, needs to have one less value than breakpoints")
    parser.add_argument("--max-iter", type=int, default=10000, help="maximum amount of iterations that a simulation can take, before it gets terminated (defaults to 10 000)")
    parser.add_argument("--iter-per-step", type=int, default=100, help="number of iterations (defaults to 100)")
    parser.add_argument("--seed", type=int, default=None, help="integer used to seed the random number generator")

    parser.add_argument("--N", type=int, default=128, help="grid size in x and y axes (defaults to 128)")
    parser.add_argument("--max-jump", type=int, default=10, help="maximum jump that a walker can make")

    parser.add_argument("--dont-show-plots", action="store_const", const=True, default=False, help="don't show plots, when simulations finish")
    parser.add_argument("--no-load", action="store_const", const=True, default=False, help="if present, script will not load previous simulations results before running new ones")

    args = parser.parse_args()

    if args.csv_dir is not None:
        directory = Path(args.csv_dir).resolve()
        if not directory.is_dir():
            raise NotADirectoryError("{} is not a directory".format((directory)))
    if args.plot_dir is not None:
        directory = Path(args.csv_dir).resolve()
        if not directory.is_dir():
            raise NotADirectoryError("{} is not a directory".format((directory)))

    if len(args.breakpoints) < 2:
        raise ValueError("At least two breakpoints need to be supplied (beginning and end)")
    if len(args.breakpoints) - len(args.step_sizes) != 1:
        raise ValueError("There needs to be exactly one less step size values supplied than breakpoints")
    
    if args.N < 1:
        raise ValueError("N needs to be bigger than one")
    if args.max_iter < 1:
        raise ValueError("max-iter needs to be bigger than one")
    if args.iter_per_step < 1:
        raise ValueError("iter-per-step needs to be bigger than one")
    if args.max_jump < 1:
        raise ValueError("max-jump needs to be bigger than one")

    return args


# %%
def main():

    args = parse_arguments()
    print(args) # TODO: delete this

    N             = args.N
    L             = args.max_jump

    breakpoints   = args.breakpoints
    steps         = args.step_sizes
    low_limit     = breakpoints[0]
    high_limit    = breakpoints[-1]

    seed          = args.seed
    max_iter      = args.max_iter
    iter_per_step = args.iter_per_step

    csv_dir = args.csv_dir
    plot_dir = args.plot_dir
    dont_show_plots = args.dont_show_plots
    no_load = args.no_load

    percents = Plot.make_plot_steps(breakpoints, steps)

    # building string for logging
    num_steps = len(percents)
    number_of_digits = lambda number: len(str(number))
    current_progress = "\r| {:>" + str(number_of_digits(num_steps)) + "}/" +\
        "{} | " +\
        "{:>" + str(number_of_digits(int(100*high_limit)) + 3) + ".2f}% | " +\
        "{:>" + str(number_of_digits(iter_per_step)) + "}/" +\
        "{:>" + str(number_of_digits(iter_per_step)) + "} |"

    # run simulations
    death_rate = Plot(percents, steps=steps, iter_per_step=iter_per_step, title="Average death rate")
    num_iter   = Plot(percents, steps=steps, iter_per_step=iter_per_step, title="Average duration")

    for iteration, percent in enumerate(percents):
        M = int(percent * N * N)

        # TODO: make a static class method for loading from csv dir
        # load simulations from csv
        num_saved = 0
        directory = Path(csv_dir).resolve()
        if csv_dir is not None and not no_load:
            csv_path = Simulation.make_file_path(directory, N, M, L)
            try:
                with csv_path.open() as sim_file:
                    for sim_line in sim_file:
                        sim_i = SimulationA(N=N, M=M, csv_line=sim_line)

                        death_rate.add_result(sim_i.ts_dead[sim_i.last_iter] / M if M != 0 else 0)
                        num_iter.add_result(sim_i.last_iter)

                        num_saved += 1
                        if num_saved == iter_per_step:
                            break
            except FileNotFoundError:
                pass

        # run the remaining simulations
        if iter_per_step > num_saved:
            sim = SimulationA(N=N, M=M, max_iter=max_iter)
            threads = cpu_count()
            with Pool(threads) as pool:
                seeds = np.full(iter_per_step - num_saved, seed)
                print(current_progress.format(iteration +  1, num_steps, percent * 100, num_saved, iter_per_step), end="")
                for sim_i in pool.imap_unordered(sim.run, seeds):
                    last_iter = sim_i.last_iter
                    #ts_sick   = sim_i.ts_sick
                    ts_dead   = sim_i.ts_dead

                    death_rate.add_result(ts_dead[last_iter] / M if M != 0 else 0)
                    num_iter.add_result(last_iter)

                    if csv_dir is not None:
                        sim_i.dump_to_csv(directory)

                    num_saved += 1
                    print(current_progress.format(iteration +  1, num_steps, percent * 100, num_saved, iter_per_step), end="")
        print(current_progress.format(iteration +  1, num_steps, percent * 100, num_saved, iter_per_step))

    if not dont_show_plots:
        death_rate.plot()
        death_rate.show()
        num_iter.plot()
        num_iter.show()

    if plot_dir is not None:
        path = Path(plot_dir).resolve()
        print(path)
        death_rate.plot()
        death_rate.save(path, "density-death-")
        num_iter.plot()
        num_iter.save(path, "density-iter-")


# %%
if __name__ == "__main__":
    main()
