import numpy as np


class PositionIndex:

    def __init__(self, grid_x, grid_y=1):
        self.__grid_x = grid_x
        self.__grid_y = grid_y
        # creates 2 dimensional array, that can store any generic objects - a bit hacky, but works
        self.__index = np.empty((grid_x, grid_y), dtype=object)
        for i in np.arange(grid_x):
            for j in np.arange(grid_y):
                # TODO: try using np.array
                self.__index[i, j] = set()
       
    def get_index(self, pos):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")

        return self.__index[pos[0], pos[1]]
    
    def append_index(self, pos, actor_idx):
        #if len(pos) != 2:
        #    raise ValueError("Position parameter needs to contain two values")

        self.__index[pos[0], pos[1]].add(actor_idx)

    def remove_index(self, pos, actor_idx):
        #if len(pos) != 2:
        #    raise ValueError("Position parameter needs to contain two values")

        self.__index[pos[0], pos[1]].remove(actor_idx)

