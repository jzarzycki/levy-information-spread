# TODO: test performance with sets vs np.arrays
# np.array - probably overall faster becasue it's implemented in C
# set - probably faster deletion by value: O(1) vs O(n) operation

import numpy as np


class PositionIndex:

    def __init__(self, grid_x: int, grid_y: int=1):
        self.__grid_x = grid_x
        self.__grid_y = grid_y
        # creates 2 dimensional array, that can store any generic objects - a bit hacky, but works
        self.__index = np.empty((grid_x, grid_y), dtype=object)
        for i in np.arange(grid_x):
            for j in np.arange(grid_y):
                # TODO: try using np.array
                self.__index[i, j] = set()

    def __getitem__(self, pos: tuple) -> set:
        if type(pos) is not tuple:
            raise ValueError
        return self.__index[pos]

    def append_index(self, pos: tuple, actor_idx: int):
        self[pos].add(actor_idx)

    def remove_index(self, pos: tuple, actor_idx: int):
        self[pos].remove(actor_idx)


class ActorIndex:

    def __init(self, M: int):
        # TODO: try using np.array
        self.__index = set()

    def __iter__(self):
        for idx in self.__index:
            yield idx

    def add(self, idx: int):
        self.__index.add(idx)

    def remove(self, idx: int):
        self.__index.remove(idx)