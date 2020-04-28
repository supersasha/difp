import numpy as np
from data import D_S0, D_S1, D_S2, SPECTRE_SIZE, ILLUM_A
import data
import colors

from utils import cv65to31

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
D50_31 = cv65to31(D50)
D55 = daylight(5500)
D55_31 = cv65to31(D55)
D65 = daylight(6500)
D65_31 = cv65to31(D65)

D50N = normalize(D50)
D50N_31 = cv65to31(D50N)
D55N = normalize(D55)
D55N_31 = cv65to31(D55N)
D65N = normalize(D65)
D65N_31 = cv65to31(D65N)

A = ILLUM_A
A_31 = cv65to31(A)
AN = normalize(A)
AN_31 = cv65to31(AN)

E = 100.0 * np.ones(SPECTRE_SIZE)
E_31 = cv65to31(E)
EN = normalize(E)
EN_31 = cv65to31(EN)

def white_point(ill_400_700):
    xyz = np.dot(data.A_1931_64_400_700_10nm.transpose(), ill_400_700)
    #print('t = ', colors.cct(xyz))
    return colors.chromaticity(xyz)

if __name__ == '__main__':
    print('White points:')
    print('A  :', white_point(to_400_700_10nm(A)))
    print('E  :', white_point(to_400_700_10nm(E)))
    print('D50:', white_point(to_400_700_10nm(D50)))
    print('D55:', white_point(to_400_700_10nm(D55)))
    print('D65:', white_point(to_400_700_10nm(D65)))
