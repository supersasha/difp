def cohen_comps_to_380_700():
    xs = np.linspace(380, 770, 40)
    xs_prim = np.linspace(380, 700, 65)
    c1 = CubicSpline(xs, data.COMP_1, bc_type='natural')
    c2 = CubicSpline(xs, data.COMP_2, bc_type='natural')
    c3 = CubicSpline(xs, data.COMP_3, bc_type='natural')
    return np.vstack([c1(xs_prim), c2(xs_prim), c3(xs_prim)])

def xyz_to_cohen_spectrum(xyz):
    def f(x):
        sp = np.dot(data.COHEN_1964.transpose(), x)
        p = 0
        for i in range(65):
            if sp[i] < 0:
                p += sp[i]*sp[i]
        d = spectrum_to_xyz(sp) - xyz
        return np.sum(d*d) + 1000000 * p
    bounds = [(-1, 1)] * 4
    res = opt.dual_annealing(f, bounds, maxiter=100)
    print(res)
    return np.dot(data.COHEN_1964.transpose(), res.x)

def xyz_to_fairman_spectrum(xyz):
    #print(data.A1931_78)
    A = (data.A1931_78[4:].transpose()).transpose()# * illum.D65N[4:]).transpose()
    Q_v0 = np.dot(A.transpose(), data.FAIRMAN_V0)
    T = np.dot(A.transpose(), data.FAIRMAN_V)
    Tinv = inv(T)
    print(xyz)
    print(Q_v0)
    C = np.dot(Tinv, xyz - Q_v0)
    print(C)
    sp = data.FAIRMAN_V0 + np.dot(data.FAIRMAN_V, C)
    print(np.dot(A.transpose(), sp))
    return sp


def color(x, y, z):
    return np.array([x, y, z])

    #spectrum = xyz_to_spectrum(np.array([0, 0, 1]))
    #xyz = spectrum_to_xyz(spectrum);
    #print(np.sum((xyz - np.array([1, 0, 0]))**2))
    #print(exposure_density(sense, spectrum))
    #xs = np.linspace(380, 700, 65)
    #print(len(data.COMP_1))
    #spectrum = xyz_to_cohen_spectrum(color(0, 1, 0))
    #print(spectrum_to_xyz(spectrum))
    #plt.plot(xs, spectrum, 'r')
    #plt.show()
    xs = np.linspace(400, 700, 61)
    sp = xyz_to_fairman_spectrum(color(290, 300, 990))
    plt.plot(xs, sp, 'r')
    #plt.plot(xs, data.FAIRMAN_V0, 'k')
    #plt.plot(xs, data.FAIRMAN_V.transpose()[0], 'r')
    #plt.plot(xs, data.FAIRMAN_V.transpose()[1], 'g')
    #plt.plot(xs, data.FAIRMAN_V.transpose()[2], 'b')
    plt.show()


    #plt.plot(data.COMP_1, 'r')
    #plt.plot(data.COMP_2, 'g')
    #plt.plot(data.COMP_3, 'b')
    #plt.plot(data.COMP_4, 'm')
    #plt.plot(data.COMP_1*0.09623 + data.COMP_2*0.09210 - data.COMP_3*0.04637 - data.COMP_4*0.05169, 'c')
    #plt.show()
