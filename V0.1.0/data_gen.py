import numpy as np
from numpy.random import randn, randint

X = np.concatenate((np.ones((100_000, 1)), randn(100_000, 11)), axis=1)
β = randint(1, 16, size=12) * randint(2, size=12)
y = X@β + randn(100_000)

X[:, 0] = y
header = ", ".join(["y"] + [f"X{i+1}" for i in range(11)])
np.savetxt("data.csv", X, delimiter=",", header=header)

print(β)