import numpy as np


def normalized(x, minor=0, major=1):
    x = np.array(x, dtype=float)
    x = (
        (
            (major-minor)*x + minor*x.max() - major*x.min()
        ) /
        (x.max() - x.min())
    )
    return x


x = np.arange(5,10, 1)
print(x)
x = normalized(x, -100, 100)
print(x)
