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
        self.plots = dict()
        self.axes = None

    def set_axes(self, axes):
        self.axes = axes

class PlotterPlot:
    def __init__(self, **params):
        self.params = params
        self.line = None

class PlotterWidget:
    def __init__(self, name, type='slider', **params):
        self.params = params
        self.name = name
        self.type = type
        self.widget = None

class Plotter:
    def __init__(self, axesShape=(1, 1), widgetsShape=(0, 0)):
        self.axesShape = axesShape
        self.widgetsShape = widgetsShape
        self.axes = dict()
        self.handlers = list()
        self.widgets = dict()

    def add_slider(self, name, ax, **params):
        self.widgets[ax] = PlotterWidget(name, type='slider', **params)

    def set_axes_params(self, ax, xs, **params):
        self.axes[ax] = PlotterAxes(xs, params)
        #{'xs': xs, 'params': params, 'plots': dict()}

    def add_plot(self, ax, name, **params):
        """
            params - like color, line width etc. if they differ from default
        """
        self.axes[ax].plots[name] = PlotterPlot(**params)

    def add_handler(self, handler):
        """
            Function handler(widgets, plots) should return
            dictionary { <plotName>: <ys> }
            for each plot which should be updated

            'widgets' are dictionary: { "name": <widgetName>, ... }
            where ... - additional widget-specific params like value for slider

            'plots' are dictionary: { "name": <xs> }
        """
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

        plots = self._get_plots()

        nrows, ncols = self.axesShape
        for i in range(nrows):
            for j in range(ncols):
                if (i, j) not in self.axes:
                    continue
                a = self.axes[(i, j)]
                axes = plt.subplot(nrows, ncols, ncols*i + j + 1)
                a.set_axes(axes)
                axes.set(**a.params)
                axes.set(xlim=(a.xdata[0], a.xdata[-1]))
                for pname in a.plots.keys():
                    line, = axes.plot(a.xdata, plots[pname], **a.plots[pname].params)
                    a.plots[pname].line = line

        plt.show()
    
    def _update(self, x):
        #print('updated: ', x)
        plots = self._get_plots()
        
        nrows, ncols = self.axesShape
        for i in range(nrows):
            for j in range(ncols):
                if (i, j) not in self.axes:
                    continue
                a = self.axes[(i, j)]
                for pname in a.plots.keys():
                    if pname in plots:
                        a.plots[pname].line.set_ydata(plots[pname])
        plt.gcf().canvas.draw_idle()

    def _setup_widgets(self):
        nrows, ncols = self.widgetsShape
        LWL = 0.05 # label width left
        LWR = 0.05 # label width right
        SH = 0.02 # slider height
        AH = 0.03 # axes height
        AW = 1 / ncols - LWL - LWR # axes width

        fc = 'lightgoldenrodyellow'


        for i in range(nrows):
            for j in range(ncols):
                idx = (i, j)
                if idx not in self.widgets:
                    continue
                pw = self.widgets[idx]
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

    def _get_plots_xs(self):
        nrows, ncols = self.axesShape
        plots_xs = {}
        for i in range(nrows):
            for j in range(ncols):
                if (i, j) not in self.axes:
                    continue
                a = self.axes[(i, j)]
                for pname in a.plots.keys():
                    plots_xs[pname] = a.xdata
        return plots_xs

    def _get_plots(self):
        plots_xs = self._get_plots_xs()
        ws = self._get_widgets()
        plots = {}
        for h in self.handlers:
            plots = {**plots, **h(ws, plots_xs)}
        return plots

    def _get_widgets(self):
        ws = {}
        nrows, ncols = self.widgetsShape
        for i in range(nrows):
            for j in range(ncols):
                idx = (i, j)
                if idx not in self.widgets:
                    continue
                w = self.widgets[idx]
                ws[w.name] = w.widget
        return DictToObject(ws)

    def _setup_rc_params(self):
        plt.rcParams['font.size'] = 5
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['grid.linewidth'] = 0.3

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()


if __name__ == '__main__':
    def handler(ws, xs):
        return {
            'plot1': xs['plot1']**(ws.r.val/10),
            'plot2': np.sin(xs['plot2']*ws.g.val)*ws.r.val,
            'plot3': np.exp(-abs(xs['plot3'])**ws.b.val)
        }
    p = Plotter((2, 2), (2, 2))

    p.set_axes_params((0, 1), np.linspace(0, 10, 1000), ylim=(0, 3.5), title='sqrt')
    p.add_plot((0, 1), 'plot1', c='r')

    p.set_axes_params((1, 0), np.linspace(-5*np.pi, 5*np.pi, 1000), ylim=(-10, 10), title='sin')
    p.add_plot((1, 0), 'plot2', c='g')
    
    p.set_axes_params((1, 1), np.linspace(-10, 10, 1000), ylim=(-2, 2), title='exp')
    p.add_plot((1, 1), 'plot3', c='b')

    p.add_handler(handler)

    p.add_slider('r', (0, 0), valmin=0, valmax=10, valstep=0.1, valinit=5)
    p.add_slider('g', (1, 0), valmin=0, valmax=10, valstep=0.1, valinit=5)
    p.add_slider('b', (0, 1), valmin=-5, valmax=5, valstep=0.1, valinit=0)
    p.show()
