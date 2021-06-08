import math
import numpy as np
import matplotlib.pyplot as plt


def levy(x, c, mi):
	upper = math.e ** -(c/(2*(x-mi)))
	lower = (x - mi) ** (3/2)
	return math.sqrt(c / (2*math.pi)) * (upper/lower)


mi = 0
c  = 1

X = np.arange(mi, 10+mi, 0.01)
y = np.array([
	levy(x, c, mi)
	for x
	in X
])

plt.title("Rozkład Lévy'ego\nc={}, µ={}".format(c, mi))
plt.xlabel("x")
plt.ylabel("Gęstość prawobodobieństwia")
plt.plot(X, y)
plt.show()