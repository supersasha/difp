import numpy as np
import matplotlib.pyplot as plt
from utils import vectorize

@vectorize(exc=[1, 2, 3, 4])
def sigma(x, _min, _max, gamma, smoo):
    a = (_max - _min) / 2
    y = gamma * (x + 0.5/gamma - 1) / a
    w = y / ((1 + abs(y)**(1/smoo))**smoo)
    return a * (w + 1)

xs = np.linspace(-5, 2, 1000)
plt.plot(xs, sigma(xs, 0, 1, 1, 0.2))
plt.plot(xs, sigma(xs, 0, 1, 0.5, 0.01))
plt.plot(xs, sigma(xs, 0, 1, 1/3, 0.01))
plt.show()

