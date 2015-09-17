from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

import palettable


def cmap_discretize(cmap, indices):
    """Discretize colormap according to indices list.

        Parameters:
        -----------
            cmap: str or Colormap instance
            indices: list

        Returns:
        --------
            segmented colormap
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    indices = np.ma.concatenate([[0], indices, [1]])
    N = len(indices)

    colors_i = np.concatenate((np.linspace(0., 1., N),
                              (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)

    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1, ki],
                       colors_rgba[i, ki]) for i in xrange(N)]

    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def get_colors():
    """Return colorbrewer's Set1 colors and a colormap similar to MATLAB's
    'parula' colormap.

        Returns:
        --------
        colors: list
        parula: list
        parula_discrete: list

    """

    colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    rgb = [[0.20784314, 0.16470588, 0.52941176],
           [0.01176471, 0.38823529, 0.88235294],
           [0.07843137, 0.52156863, 0.83137255],
           [0.02352941, 0.65490196, 0.77647059],
           [0.21960784, 0.72549020, 0.61960784],
           [0.57254902, 0.74901961, 0.45098039],
           [0.85098039, 0.72941176, 0.33725490],
           [0.98823529, 0.80784314, 0.18039216],
           [0.97647059, 0.98431373, 0.05490196]]
    parula = LinearSegmentedColormap.from_list('parula', rgb, N=256)
    parula_discrete = ListedColormap(rgb, name='parula_discrete', N=9)

    for cmap in (parula, parula_discrete):
        matplotlib.cm.register_cmap(cmap=cmap)

    return colors, parula, parula_discrete

def get_defaults():
    """Set better font sizes, label sizes and line widths."""
    font = {'size': 18,
            'family': 'Times New Roman'}
    matplotlib.rc('font', **font)
    axes = {'labelsize': 20}
    matplotlib.rc('axes', **axes)
    lines = {'linewidth': 2.5}
    matplotlib.rc('lines', **lines)


if __name__ == '__main__':
    pass
