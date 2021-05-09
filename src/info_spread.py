#!/usr/bin/env python3

"""
Information spread model using Lévy Flight random walks.
"""

from pathlib import Path
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from index import PositionIndex
    from random_walks import Levy
else:
    from src.index import PositionIndex
    from src.random_walks import Levy


class Simulation(ABC):

    SUSCEPTIBLE = 0
    INFECTED    = 1
    RECOVERED   = 2


    def __init__(self, N: int=None, M: int=None, max_random_step: int=10,
                    max_iter: int=10000, logging: bool=False, csv_line: str=None):
        self.N = N
        self.M = M

        self.max_random_step = max_random_step
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


    @abstractmethod
    def handle_two_infected_meet(self) -> tuple:
        pass


    def run(self, seed: int):
        if seed is not None:
            np.random.seed(seed)

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
                    new_x = (x[j] + self.random_walk.random_step()) % self.N
                    new_y = (y[j] + self.random_walk.random_step()) % self.N
                    x[j] = new_x
                    y[j] = new_y
                    position_index.append_index((new_x, new_y), j)

                    # save which walkers changed status
                    same_position = position_index[(x[j], y[j])]
                    for k in same_position:
                        # if an infected person meets a susceptible person, infect them
                        if infect[k] == self.SUSCEPTIBLE and j != k:
                            new_infected.add(k)
                        # if an infected person meets a recovered person, they both become recovered
                        elif infect[k] == self.RECOVERED and j != k:
                            new_recovered.add(j)
                        # if two infected people meet both/one of them become recovered (depends on model)
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
                infected_index.remove(idx)
            new_recovered.clear()

            # plot data
            self.ts_sick[iteration] = n_sick
            self.ts_dead[iteration] = n_dead
            iteration += 1

            if self.logging:
                print(f"I:{iteration}, healthy: {self.M - n_sick - n_dead}, sick:{n_sick}, dead:{n_dead}")

        self.last_iter = iteration - 1
        return self


    def plot(self):
        plt.figure()
        plt.title("Information spread simulation\nN:{}, M:{}, L:{}".format(self.N, self.M, self.max_random_step))
        plt.xlabel("Number of iterations")
        plt.ylabel("% of affected population")
        plt.plot(np.arange(self.last_iter), self.ts_sick[0:self.last_iter]/self.M * 100) # percent


    def show(self):
        plt.show()


    def save(self, path: Path, filename: str):
        files = Path().glob(filename + "*.png")
        num = len(list(files)) + 1
        file_path = path.joinpath(filename+ str(num) + ".png")
        plt.savefig(file_path)

    
    @staticmethod
    def make_file_path(directory, N, M, max_step):
        return Path(directory).joinpath("{}N-{}M-{}L.csv".format(N, M, max_step))


    def dump_to_csv(self, directory: Path):
        file_path = self.make_file_path(directory, self.N, self.M, self.max_random_step)
        with file_path.open(mode="a") as f:

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


class SimulationA(Simulation):

    def handle_two_infected_meet(self) -> tuple:
        return self.RECOVERED, self.RECOVERED


class SimulationB(Simulation):

    def handle_two_infected_meet(self) -> tuple:
        return self.INFECTED, self.RECOVERED


def parse_arguments():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--csv-dir", default=None, help="path to directory, where simulation results are to be kept")
    parser.add_argument("--N", type=int, nargs="?", default=128, help="grid size in x and y axes (defaults to 128)")
    parser.add_argument("--ro", type=float, nargs="?", default=0.1, help="population size (defaults to 0.1)")
    parser.add_argument("--max-random-step", type=int, default=10, help="maximum step size, that an actor can make")
    parser.add_argument("--load", action="store_const", const=True, default=False, help="if present, script will load previous simulations results instead of running new ones")

    args = parser.parse_args()

    if args.csv_dir is not None:
        directory = Path(args.csv_dir).resolve()
        if not directory.is_dir():
            raise NotADirectoryError("{} is not a directory".format((directory)))
    elif args.load:
        raise ValueError("Supply a directory with csv results with the --csv_dir flag")

    if args.ro < 0:
        raise ValueError("Population density can't be lower than zero")
    if args.max_random_step < 1:
        raise ValueError("Max step size can't be lower than one")

    return args


def main():
    args = parse_arguments()

    if args.csv_dir is not None:
        directory = Path(args.csv_dir).resolve()
    N = args.N
    M = int(args.ro * N**2)
    L = args.max_random_step

    if args.load:
        file_name = self.make_file_path(directory, N, M, self.max_random_step)

        try:
            with directory.joinpath(file_name).open() as f:
                for line in f:
                    simulation = SimulationA(N=N, M=M, csv_line=line)
                    simulation.plot()
                    simulation.show()
        except FileNotFoundError:
            print("There aren't any simulations of this kind yet.")

    else:
        simulation = SimulationA(N=N, M=M, max_random_step=L, max_iter=10000, logging=True)
        simulation.run(123)

        if args.csv_dir is not None:
            simulation.dump_to_csv(directory)

        simulation.plot()
        simulation.show()


if __name__ == "__main__":
    main()
