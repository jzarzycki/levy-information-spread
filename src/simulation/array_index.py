import numpy as np


class PositionIndex:

    def __init__(self, grid_x, grid_y=1):
        self.__grid_x = grid_x
        self.__grid_y = grid_y
        # creates 2 dimensional array, that can store any generic objects - a bit hacky, but works
        self.__index = np.empty((grid_x, grid_y), dtype=object)
       
    def get_index(self, pos):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")

        return self.__index[pos[0], pos[1]]
    
    def append_index(self, pos, actor_idx):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")

        idxs = self.__index[pos[0], pos[1]]
        if idxs is None:
            idxs = [actor_idx] # TODO: try with numpy arrays?
        else:
            idxs.append(actor_idx)

        self.__index[pos[0], pos[1]] = idxs

    def remove_index(self, pos, actor_idx):
        if len(pos) != 2:
            raise ValueError("Position parameter needs to contain two values")

        self.__index[pos[0], pos[1]].remove(actor_idx)

