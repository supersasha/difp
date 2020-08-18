import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    #plt.style.use('seaborn')

    plt.rcParams['font.size'] = 5
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['toolbar'] = 'None'
    plt.rcParams['axes.grid'] = 'True'
    plt.rcParams['grid.linewidth'] = 0.3
    #plt.rcParams['figure.autolayout'] = 'True'

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.7, wspace=0.15, hspace=0.2)

    xs = np.linspace(-10, 10, 1000)
    plt.subplot(231)
    plt.plot(xs, np.sin(xs))
    plt.subplot(232)
    plt.plot(xs, np.sin(np.exp(xs)))
    plt.subplot(233)
    plt.gca().set_ylim(-10, 10)
    plt.plot(xs, np.tan(xs))
    plt.subplot(234)
    plt.plot(xs, np.sin(np.exp(xs)))
    plt.subplot(235)
    plt.plot(xs, np.sin(np.exp(xs)))
    plt.subplot(236)
    plt.plot(xs, np.sin(np.exp(xs)))
    #plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


