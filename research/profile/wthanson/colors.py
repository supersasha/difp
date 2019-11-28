import numpy as np
import math
import data

def color(x, y, z):
    return np.array([x, y, z])

## Color conversions

def srgb_to_xyz(c):
    r = c[0]
    g = c[1]
    b = c[2]

    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r /= 12.92;

    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g /= 12.92;

    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b /= 12.92;

    r *= 100
    g *= 100
    b *= 100

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    return color(x, y, z)

def xyz_to_srgb(c):
    x = c[0] / 100.0
    y = c[1] / 100.0
    z = c[2] / 100.0

    r = x *  3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y *  1.8758 + z *  0.0415
    b = x *  0.0557 + y * -0.2040 + z *  1.0570

    if r > 0.0031308:
        r = 1.055 * r ** (1.0 / 2.4) - 0.055
    else:
        r = 12.92 * r

    if g > 0.0031308:
        g = 1.055 * g ** (1.0 / 2.4) - 0.055
    else:
        g = 12.92 * g

    if b > 0.0031308:
        b = 1.055 * b ** (1.0 / 2.4) - 0.055
    else:
        b = 12.92 * b

    return np.clip(color(r, g, b), 0, 1)

ref_x = 95.047
ref_y = 100.0
ref_z = 108.883

def lab_to_xyz(c):

    y = (c[0] + 16.0) / 116.0
    x = c[1] / 500.0 + y
    z = y - c[2] / 200.0

    if y > 0.206893034:
        y = y * y * y
    else:
        y = (y - 16.0/116.0) / 7.787;
    
    if x > 0.206893034:
        x = x * x * x
    else:
        x = (x - 16.0/116.0) / 7.787
    
    if z > 0.206893034:
        z = z * z * z
    else:
        z = (z - 16.0/116.0) / 7.787
    
    return color(x * ref_x, y * ref_y, z * ref_z)

def xyz_to_lab(c):
    x = c[0] / ref_x
    y = c[1] / ref_y
    z = c[2] / ref_z

    if x > 0.008856:
        x = x ** (1.0/3.0)
    else:
        x = 7.787 * x + 16.0/116.0

    if y > 0.008856:
        y = y ** (1.0/3.0);
    else:
        y = 7.787 * y + 16.0/116.0

    if z > 0.008856:
        z = z ** (1.0/3.0)
    else:
        z = 7.787 * z + 16.0/116.0

    return color(116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z))

#def xyz_to_lab_vec(c):
#    u = c / np.array([ref_x, ref_y, ref_z])
#    np.where(u[:, 0] > ..., )

def lab_to_lch(c):
    return color(c[0], math.hypot(c[1], c[2]), math.atan2(c[2], c[1]))

def xyz_to_lch(c):
    return lab_to_lch(xyz_to_lab(c))

## Color Distance

def delta_E76_lab(lab1, lab2):
    '''
        Minimum noticible distance is 2.3
    '''
    d = lab1 - lab2
    return np.sqrt(np.sum(d*d))

def delta_E76_xyz(xyz1, xyz2):
    return delta_E76_lab(xyz_to_lab(xyz1), xyz_to_lab(xyz2))

def delta_E94_xyz(xyz1, xyz2):
    lch1 = xyz_to_lch(xyz1)
    lch2 = xyz_to_lch(xyz2)
    KL = 1
    K1 = 0.045
    K2 = 0.015
    d1 = (lch2[0] - lch1[0]) / KL
    d2 = (lch2[1] - lch1[1]) / (1 + K1*lch1[1])
    d3 = (lch2[2] - lch1[2]) / (1 + K2*lch1[1]) # <- K2*lch1[1] !
    return math.sqrt(d1*d1 + d2*d2 + d3*d3)

def chromaticity(xyz):
    v = np.sum(xyz)
    if v == 0:
        return np.array([1/3, 1/3])
    return np.array([xyz[0] / v, xyz[1] / v])

def cct(xyz):
    '''
        Color temperature
        Use for white points only
    '''
    [x, y] = chromaticity(xyz)
    xe = 0.3320
    ye = 0.1858
    n = (x - xe)/(y - ye)
    return ((-449 * n + 3525) * n - 6823.3) * n + 5520.33

def spectral_color_xy(lam):
    idx = int((lam - 380) / 5)
    c = data.A1931_78[idx]
    s = np.sum(c)
    return np.array([c[0]/s, c[1]/s])

def xyY_to_XYZ(x, y, Y):
    return color(x*Y/y, Y, (1-x-y)*Y/y)
