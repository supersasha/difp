import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator

if __name__ == '__main__':
    intrp = PchipInterpolator
    xs = np.linspace(400, 700, 31)
    spl1 = intrp([350, 440, 560, 680, 750], [0, 0.498, 0.552, 0, 0])
    plt.plot(xs, spl1(xs))
    spl2 = intrp([350, 440, 560, 680, 750], [0, 0, 0.351, 0, 0])
    plt.plot(xs, spl2(xs))
    spl3 = intrp([350, 440, 560, 680, 750], [0, 0.343, 0, 0.467, 0])
    plt.plot(xs, spl3(xs))
    plt.plot(xs, spl1(xs) + spl2(xs) + spl3(xs), 'k')
    plt.show()
