from utils import vectorize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import math

@vectorize()
def sigma(x, ymin, ymax, gamma, bias, smoo):
    a = (ymax - ymin) / 2
    y = gamma * (x - bias) / a
    return a * (y / (1 + abs(y)**(1/smoo))**smoo + 1) + ymin

def sigma_from(x, ymin, ymax, gamma, smoo, x0):
    avg = (ymax + ymin) / 2

    # gamma * (x0 - bias) + avg = ymin
    
    bias = x0 - (ymin - avg) / gamma
    return sigma(x, ymin, ymax, gamma, bias, smoo)

def sigma_to(x, ymin, ymax, gamma, smoo, x1):
    avg = (ymax + ymin) / 2

    # gamma * (x1 - bias) + avg = ymax
    
    bias = x1 - (ymax - avg) / gamma
    return sigma(x, ymin, ymax, gamma, bias, smoo)

@vectorize()
def zigzag(x, ymin, ymax, gamma, bias):
    if x <= bias:
        return ymin
    y = ymin + gamma * (x - bias)
    if y > ymax:
        return ymax
    return y

def main_sigma():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.5)
    xs = np.linspace(-5, 5, 1000)
    mn = -1
    mx = 1
    gamma = 1
    bias = 0
    smoo = 1
    ys = sigma(xs, mn, mx, gamma, bias, smoo)
    d = mx - mn
    dd = (mx + mn) / 2
    ax.set_ylim(mn - 0.1*d, mx + 0.1*d)
    #ax.set_ylim(-5, 5)
    l, = plt.plot(xs, ys)
    l1, = plt.plot(xs, gamma*(xs-bias)+dd)

    axMn = plt.axes([0.25, 0.4, 0.65, 0.03])
    axMx = plt.axes([0.25, 0.35, 0.65, 0.03])
    axGamma = plt.axes([0.25, 0.30, 0.65, 0.03])
    axBias = plt.axes([0.25, 0.25, 0.65, 0.03])
    axSmoo = plt.axes([0.25, 0.20, 0.65, 0.03])

    slMn = Slider(axMn, 'Min', -1, 0.4, valinit=mn, valstep=0.1)
    slMx = Slider(axMx, 'Max', 0.6, 2, valinit=mx, valstep=0.1)
    slGamma = Slider(axGamma, 'Gamma', -10, 10, valinit=gamma, valstep=0.1)
    slBias = Slider(axBias, 'Bias', -5, 5, valinit=bias, valstep=0.1)
    slSmoo = Slider(axSmoo, 'Smoothing', 0.01, 2, valinit=smoo, valstep=0.01)

    def update(val):
        mn = slMn.val
        mx = slMx.val
        gamma = slGamma.val
        bias = slBias.val
        smoo = slSmoo.val
        ys = sigma(xs, mn, mx, gamma, bias, smoo)
        d = mx - mn
        dd = (mx + mn) / 2
        ax.set_ylim(mn - 0.1*d, mx + 0.1*d)
        l.set_ydata(ys)
        l1.set_ydata(gamma*(xs-bias) + dd)
        fig.canvas.draw_idle()

    slMn.on_changed(update)
    slMx.on_changed(update)
    slGamma.on_changed(update)
    slBias.on_changed(update)
    slSmoo.on_changed(update)

    plt.show()

def main_zigzag():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.5)
    xs = np.linspace(-5.0, 5.0, 1000)
    mn = -1
    mx = 1
    gamma = 1
    bias = 0
    ys = zigzag(xs, mn, mx, gamma, bias)
    d = mx - mn
    dd = (mx + mn) / 2
    ax.set_ylim(mn - 0.1*d, mx + 0.1*d)
    #ax.set_ylim(-5, 5)
    l, = plt.plot(xs, ys)
    l1, = plt.plot(xs, gamma*(xs-bias)+dd)

    axMn = plt.axes([0.25, 0.4, 0.65, 0.03])
    axMx = plt.axes([0.25, 0.35, 0.65, 0.03])
    axGamma = plt.axes([0.25, 0.30, 0.65, 0.03])
    axBias = plt.axes([0.25, 0.25, 0.65, 0.03])

    slMn = Slider(axMn, 'Min', -2, 0.4, valinit=mn, valstep=0.01)
    slMx = Slider(axMx, 'Max', 0.6, 2, valinit=mx, valstep=0.1)
    slGamma = Slider(axGamma, 'Gamma', -10, 10, valinit=gamma, valstep=0.1)
    slBias = Slider(axBias, 'Bias', -5, 5, valinit=bias, valstep=0.1)

    def update(val):
        mn = slMn.val
        mx = slMx.val
        gamma = slGamma.val
        bias = slBias.val
        ys = zigzag(xs, mn, mx, gamma, bias)
        d = mx - mn
        dd = (mx + mn) / 2
        ax.set_ylim(mn - 0.1*d, mx + 0.1*d)
        l.set_ydata(ys)
        l1.set_ydata(gamma*(xs-bias) + dd)
        fig.canvas.draw_idle()

    slMn.on_changed(update)
    slMx.on_changed(update)
    slGamma.on_changed(update)
    slBias.on_changed(update)

    plt.show()

def main2():
    xs = np.linspace(-10, 10, 1000)
    plt.plot(xs, sigma_from(xs, 0, 4, 1, 0.1, 5), 'b')
    plt.plot(xs, sigma_to  (xs, 0, 4, 1, 0.1, 5), 'g')
    plt.show()

if __name__ == '__main__':
    main_zigzag()

