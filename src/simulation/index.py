import numpy as np


class PositionIndex:

    def __init__(self, grid_x: int, grid_y: int=1):
        self.__grid_x = grid_x
        self.__grid_y = grid_y
        # creates 2 dimensional array, that can store any generic objects - a bit hacky, but works
        self.__index = np.empty((grid_x, grid_y), dtype=object)
        for i in np.arange(grid_x):
            for j in np.arange(grid_y):
                self.__index[i, j] = set()

    def __getitem__(self, pos: tuple) -> set:
        if type(pos) is not tuple:
            raise ValueError
        return self.__index[pos]

    def append_index(self, pos: tuple, actor_idx: int):
        self[pos].add(actor_idx)

    def remove_index(self, pos: tuple, actor_idx: int):
        self[pos].remove(actor_idx)