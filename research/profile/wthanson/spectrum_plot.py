import numpy as np

from plotter import Plotter
import spectrum
import colors
import illum
from utils import vectorize
import matplotlib.cm as cm

@vectorize(exc=[3, 4])
def spectrum_dist(x, y, Y, refl_gen, mtx_refl):
    xyz = colors.xyY_to_XYZ(x, y, Y)
    sp, refl = refl_gen.spectrum_of(xyz)
    xyz1 = spectrum.transmittance_to_xyz(mtx_refl, refl)
    return colors.delta_E94_xyz(xyz, xyz1)

def main():
    refl_gen = spectrum.load_spectrum('spectra2/spectrum-d55-4.json')
    mtx_refl = spectrum.transmittance_to_xyz_mtx(illum.D55_31)

    p = Plotter((2, 2), (2, 3))

    p.add_slider('r', (0, 0), valmin=0., valmax=1., valstep=0.01, valinit=0)
    p.add_slider('g', (0, 1), valmin=0., valmax=1., valstep=0.01, valinit=0)
    p.add_slider('b', (0, 2), valmin=0., valmax=1., valstep=0.01, valinit=0)

    p.add_slider('Y', (1, 0), valmin=0., valmax=100., valstep=1., valinit=80.)

    p.add_axes((0, 0), None)
    p.add_image('sRGB', (0, 0), [[[0., 0., 0.], [0., 0., 0.]]])

    p.add_axes((0, 1), np.linspace(400, 700, 31), ylim=(-0.1, 1.5))
    p.add_plot('uspectrum', (0, 1), ls='--')
    p.add_plot('spectrum', (0, 1))

    x = np.linspace(0.01, 1, 30)
    X, Y = np.meshgrid(x, x)
    p.add_axes((1, 0), None)
    p.add_contour('contour', (1, 0), X, Y, cmap=cm.coolwarm, levels=[1, 3, 10])

    def handler(plotter):
        r = plotter.widget('r').widget.val
        g = plotter.widget('g').widget.val
        b = plotter.widget('b').widget.val
        y = plotter.widget('Y').widget.val

        spectrum_out = plotter.output('spectrum')
        uspectrum_out = plotter.output('uspectrum')
        xyz = colors.srgb_to_xyz(colors.color(r, g, b))
        sp, refl = refl_gen.spectrum_of(xyz)
        usp, urefl = refl_gen.unclipped_spectrum_of(xyz)
        xyz1 = spectrum.transmittance_to_xyz(mtx_refl, refl)
        rgb1 = colors.xyz_to_srgb(xyz1)
        spectrum_out.set_data(refl)
        uspectrum_out.set_data(urefl)
        print(colors.delta_E94_xyz(xyz, xyz1))

        srgb_out = plotter.output('sRGB')
        srgb_out.set_data([[[r, g, b], rgb1]])

        cont_out = plotter.output('contour')
        cont_out.set_data(spectrum_dist(X, Y, y, refl_gen, mtx_refl))

    p.add_handler(handler)

    p.show()

if __name__ == '__main__':
    main()
