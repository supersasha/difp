import numpy as np

#points = np.array([[0, 0], [0, 1.5], [0, 2], [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1.5], [2, 2]])

points = np.random.rand(10000, 2)
#points = np.random.poisson(lam=1000000, size=(1000, 2))
#points = np.random.uniform(size=(1000, 2))
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)

import matplotlib.pyplot as plt
voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors="rgbcmy", line_width=0.2)
plt.show()

