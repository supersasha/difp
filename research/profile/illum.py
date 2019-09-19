import numpy as np
from data import D_S0, D_S1, D_S2, SPECTRE_SIZE, ILLUM_A

def daylight(temp):
    x = 0
    s = 1000 / temp
    if temp <= 7000:
        x = (((-4.607 * s) + 2.9678) * s + 0.09911) * s + 0.244063
    else:
        x = (((-2.0064 * s) + 1.9018) * s + 0.24748) * s + 0.237040
    y = (-3.0 * x + 2.870) * x - 0.275;
    m = 0.0241 + 0.2562 * x - 0.7341 * y;

    m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m;
    m2 = (0.030 - 31.4424 * x + 30.0717 * y) / m;

    return D_S0 + m1 * D_S1 + m2 * D_S2

def mono(i):
    z = np.zeros(SPECTRE_SIZE)
    z[i] = 1.0
    return z

def normalize(ill):
    return ill / np.sum(ill)

D50 = daylight(5000)
D55 = daylight(5500)
D65 = daylight(6500)

D50N = normalize(D50)
D55N = normalize(D55)
D65N = normalize(D65)

A = ILLUM_A
AN = normalize(A)

E = 100.0 * np.ones(SPECTRE_SIZE)
EN = normalize(E)
