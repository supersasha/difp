import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider

class DictToObject(object):

    def __init__(self, dictionary):
        def _traverse(key, element):
            if isinstance(element, dict):
                return key, DictToObject(element)
            else:
                return key, element

        objd = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(objd)

class PlotterAxes:
    def __init__(self, xs, params):
        self.xdata = xs
        self.params = params
        #self.plots = dict()
        self.axes = None

    def set_axes(self, axes):
        self.axes = axes

class PlotterPlot:
    def __init__(self, name, axix, xdata, **params):
        self.name = name
        self.axix = axix
        self.params = params
        self.xdata = xdata
        self.line = None
        self.data = None

    def set_data(self, data):
        self.data = data

    def install(self, ax):
        self.line, = ax.axes.plot(ax.xdata, self.data, **self.params)

    def xs(self):
        return self.xdata

    def update(self):
        self.line.set_ydata(self.data)

class PlotterImage:
    def __init__(self, name, axix, data, **params):
        self.name = name
        self.axix = axix
        self.data = data
        self.params = params

    def set_data(self, data):
        self.data = data

    def install(self, ax):
        self.image = ax.axes.imshow(self.data, **self.params)
        ax.axes.grid(False)

    def update(self):
        self.image.set_data(self.data)

class PlotterContour:
    def __init__(self, name, axix, meshX, meshY, **params):
        self.name = name
        self.axix = axix
        self.meshX = meshX
        self.meshY = meshY
        self.data = None
        self.params = params

    def set_data(self, data):
        self.data = data

    def install(self, ax):
        self.ax = ax
        self.contour = ax.axes.contour(self.meshX, self.meshY, self.data,
                                                    #cmap=cm.coolwarm,
                                                    **self.params)
        ax.axes.clabel(self.contour, inline=1, fontsize=5)
    def update(self):
        self.ax.axes.cla()
        self.install(self.ax)
        #self.contour.set_array(self.data)

class PlotterWidget:
    def __init__(self, name, axix, type='slider', **params):
        self.params = params
        self.name = name
        self.axix = axix
        self.type = type
        self.widget = None

class Plotter:
    def __init__(self, axesShape=(1, 1), widgetsShape=(0, 0)):
        self.axesShape = axesShape
        self.widgetsShape = widgetsShape
        self.axes = dict()
        self.handlers = list()
        self.widgets = dict()
        self.outputs = dict()

    def widget(self, name):
        return self.widgets[name]

    def output(self, name):
        return self.outputs[name]

    def add_slider(self, name, axix, **params):
        self.widgets[name] = PlotterWidget(name, axix, type='slider', **params)

    def add_axes(self, axix, xs, **params):
        self.axes[axix] = PlotterAxes(xs, params)

    def add_plot(self, name, axix, **params):
        self.outputs[name] = PlotterPlot(name, axix, self.axes[axix].xdata, **params)

    def add_image(self, name, axix, data, **params):
        self.outputs[name] = PlotterImage(name, axix, data, **params)

    def add_contour(self, name, axix, meshX, meshY, **params):
        self.outputs[name] = PlotterContour(name, axix, meshX, meshY, **params)

    def add_handler(self, handler):
        self.handlers.append(handler)

    def show(self):
        self._setup_rc_params()
        wh = self._setup_widgets()
        plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.99,
            top=1-wh-0.05, #(0.95 - 0.05*self.widgetsShape[1]),
            wspace=0.15,
            hspace=0.2)

        for h in self.handlers:
            h(self)

        nrows, ncols = self.axesShape
        for i in range(nrows):
            for j in range(ncols):
                axix = (i, j)
                if axix not in self.axes:
                    continue
                a = self.axes[axix]
                axes = plt.subplot(nrows, ncols, ncols*i + j + 1)
                a.set_axes(axes)
                axes.set(**a.params)
                if a.xdata is not None:
                    axes.set(xlim=(a.xdata[0], a.xdata[-1]))

        for name, out in self.outputs.items():
            axes = self.axes[out.axix]
            out.install(axes)

        plt.show()
    
    def _update(self, x):
        for h in self.handlers:
            h(self)
        for out in self.outputs.values():
            out.update()
        plt.gcf().canvas.draw_idle()

    def _setup_widgets(self):
        nrows, ncols = self.widgetsShape
        LWL = 0.05 # label width left
        LWR = 0.05 # label width right
        SH = 0.02 # slider height
        AH = 0.03 # axes height
        AW = 1 / ncols - LWL - LWR # axes width

        fc = 'lightgoldenrodyellow'

        for name, pw in self.widgets.items():
            i, j = pw.axix
            axes = plt.axes([
                j * (LWL + LWR + AW) + LWL,
                1 - AH*(i+1),
                AW,
                SH], fc=fc)
            p = {**pw.params}
            label = pw.name
            if 'label' in pw.params:
                label = pw.params['label']
                del p['label']

            valmin = 0
            if 'valmin' in pw.params:
                valmin = pw.params['valmin']
                del p['valmin']

            valmax = 1
            if 'valmax' in pw.params:
                valmax = pw.params['valmax']
                del p['valmax']
           
            pw.widget = Slider(axes, label, valmin, valmax, **p)
            pw.widget.on_changed(self._update)

        return nrows * AH

    def _setup_rc_params(self):
        plt.rcParams['font.size'] = 5
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['grid.linewidth'] = 0.3

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()


if __name__ == '__main__':
    def handler(plotter):
        r = plotter.widget('r').widget.val
        g = plotter.widget('g').widget.val
        b = plotter.widget('b').widget.val
        
        p1 = plotter.output('plot1')
        p1.set_data(p1.xs() ** (r / 10))
        
        p2 = plotter.output('plot2')
        p2.set_data(np.sin(p2.xs() * g) * r)

        p3 = plotter.output('plot3')
        p3.set_data(np.exp(-abs(p3.xs())**b))

        im = plotter.output('img1')
        im.set_data([[[r, g, b], [1-r, 1-g, 1-b], [1, 1, 1]]])


    p = Plotter((2, 2), (2, 2))

    p.add_axes((0, 1), np.linspace(0, 10, 1000), ylim=(0, 3.5), title='sqrt')
    p.add_plot('plot1', (0, 1), c='r')

    p.add_axes((1, 0), np.linspace(-5*np.pi, 5*np.pi, 1000), ylim=(-10, 10), title='sin')
    p.add_plot('plot2', (1, 0), c='g')
    
    p.add_axes((1, 1), np.linspace(-10, 10, 1000), ylim=(-2, 2), title='exp')
    p.add_plot('plot3', (1, 1), c='b')

    p.add_axes((0, 0), None)
    p.add_image('img1', (0, 0), [[[1., 0, 0], [0, 1, 0], [0, 0, 1]]])

    p.add_handler(handler)

    p.add_slider('r', (0, 0), valmin=0., valmax=1., valstep=0.01, valinit=0)
    p.add_slider('g', (1, 0), valmin=0., valmax=1., valstep=0.01, valinit=0)
    p.add_slider('b', (0, 1), valmin=0., valmax=1., valstep=0.01, valinit=0)
    p.show()
