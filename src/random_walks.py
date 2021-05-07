import numpy as np
import matplotlib.pyplot as plt


class Levy:

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

    plt.title("Distribution of random values from 1 to {}".format(max_val))
    plt.xlabel("Value")
    plt.ylabel("Number of occurrences")
    plt.plot(np.arange(1, max_val + 1), count)
    plt.show()