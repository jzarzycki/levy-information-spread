#!/usr/bin/env python3

"""
Module for plotting results of running simulations for multiple values of population density in an information spread model
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class Plot:

    def __init__(self, N: int, L: int, percents: list, steps: list, iter_per_step: int, title: str, brownian: bool=False):
        self.N = N
        self.L = L
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
        if type(self.random_walk) is Levy:
            walk_name = "Levy flight"
        elif type(self.random_walk) is Brownian:
            walk_name = "Brownian Motion"
        else:
            raise NotImplementedError
        plt.figure()
        plt.title("Population density plot for an information spread model\n{}, N:{}, L:{}".format(walk_name, self.N, self.L))
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