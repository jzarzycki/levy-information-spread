#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N = 128     # grid size
M = 4000    # population size
L = 20      # lifespan
MAX_ITER = 10000

# possible movements
X_STEP = np.array([-1, 0, 1, 0])
Y_STEP = np.array([0, -1, 0, 1])

# "infect" column possible states (order inversed for performance reasons)
DEAD = 0
INFECTED = 1
HEALTHY = 2

# population information
# TODO: Is idx column still necessary?
dtype = [
    # ("idx", int),
    ("x", int),
    ("y", int),
    ("infect", int),
    ("lifespan", int)
]

population = np.array([
    (
        # 0, # idx
        0, # x
        0, # y
        HEALTHY, # infect
        0 # lifespan
    )
    for _
    in np.arange(M)
], dtype=dtype)

# population initialization
for j in np.arange(M):
    # population[j]["idx"] = j
    population[j]["x"] = np.random.randint(0, N)
    population[j]["y"] = np.random.randint(0, N)
    population[j]["lifespan"] = L

# creating the first infected
jj = np.random.randint(0, M-1)
population[jj]["infect"] = INFECTED

# plot data
n_sick, n_dead = 1, 0
ts_sick = np.zeros(MAX_ITER)

# main loop
iteration = 0
while (n_sick > 0) and (iteration < MAX_ITER):
    # move all actors
    for person in population:
        if person["infect"] != DEAD:
            ii = np.random.randint(4)
            person["x"] += X_STEP[ii]
            person["y"] += Y_STEP[ii]
            # why was the max here?
            person["x"] = min(N, max(person["x"], 1))
            person["y"] = min(N, max(person["y"], 1))

    # deal with infected - sorting version
    # TODO: Try removing dead population to speed up later iterations

    infected_coords = None

    # TODO: Research optimal sorting algorithm (might be different for levy flights)
    # from what I remember quick sort tends to be slow with pre-sorted data, so we could improve here
    population.sort(order=["x", "y", "infect"])
    for person in population:
        if person["infect"] == DEAD:
            continue

        elif person["infect"] == INFECTED:
            infected_coords = (person["x"], person["y"])
            person["lifespan"] -= 1

            if person["lifespan"] <= 0:
                person["infect"] = DEAD
                n_sick -= 1
                n_dead += 1

        elif person["infect"] == HEALTHY:
            if (person["x"], person["y"]) == infected_coords:
                person["infect"] = INFECTED
                person["lifespan"] = L
                n_sick += 1
    
    # save plot data
    ts_sick[iteration] = n_sick
    iteration += 1
    print(f"I:{iteration}, sick:{round(n_sick / M * 100, 2)}%, dead:{round(n_dead / M * 100, 2)}%.")

plt.plot(np.arange(iteration + 10), ts_sick[0:iteration + 10]/M * 100)
plt.show()
