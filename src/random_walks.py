from abc import ABC, abstractmethod
from typing import overload
import numpy as np
import matplotlib.pyplot as plt


class RandomWalk(ABC):

    @abstractmethod
    def random_step(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


class Levy(RandomWalk):

    def __init__(self, max_val):
        self.__max_val = max_val

        one_over_i = np.array([
            1/i
            for i
            in np.arange(1, max_val + 1)
        ])
        norm = 1 / sum(one_over_i)
        probability = norm * one_over_i

        self.cumultative_prob = np.zeros(max_val)
        self.cumultative_prob[0] = probability[0]
        for idx in np.arange(1, max_val):
            self.cumultative_prob[idx] = self.cumultative_prob[idx - 1] + probability[idx]


    def random_step(self):
        direction = np.random.choice([-1, 1])
        r = np.random.random_sample()
        for i, threshold in enumerate(self.cumultative_prob):
            if r <  threshold:
                return direction * (i + 1)
        return direction * self.__max_val


    def get_name(self):
        return "levy"


class Brownian(RandomWalk):

    def __init__(self, step_size=1):
        self.step_size = step_size

    def random_step(self):
        direction = np.random.choice([-1, 1])
        return direction * self.step_size

    def get_name(self):
        return "brown"


if __name__ == '__main__':
    N = 10 ** 5
    max_val = 10

    levy = Levy(max_val)
    random_steps = np.array([
        levy.random_step()
        for _ in
        np.arange(N)
    ])

    count = np.zeros(max_val)
    for value in random_steps:
        count[abs(value) - 1] += 1

    plt.title("Rozkład wylosowanych wartości dla max_val={}".format(max_val))
    plt.xlabel("Wartości")
    plt.ylabel("Ilość wystąpień")
    plt.plot(np.arange(1, max_val + 1), count)
    plt.show()

    # TODO: add plot of Gaussian distribution in Brownian motion in one axis