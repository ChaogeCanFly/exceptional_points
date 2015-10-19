#!/usr/bin/env python

from __future__ import division

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mayavi import mlab
import numpy as np

import argh

from ep.waveguide import DirichletReduced, DirichletPositionDependentLossReduced
from ep.plot import get_colors, get_defaults

colors, cmap, _ = get_colors()
get_defaults()


def plot_spectrum(eps_min=-0.01, eps_max=0.11, eps_N=101, delta_N=101,
                  pos_dep=False, show=False):

    wg_kwargs = dict(N=2.05,
            x_R0=0.1,
            y_R0=0.85,
            switch_losses_on_off=True,
            loop_type='Bell')

    if pos_dep:
        wg_kwargs.update(dict(init_phase=0.0,
                              eta=1.0,
                              eta0=1.0))
        D = DirichletPositionDependentLossReduced(**wg_kwargs)
        delta_min, delta_max = -1.1, 1.1
        # delta_min, delta_max = -0.4, 0.1
        # delta_min, delta_max = -1.15, -0.70
    else:
        wg_kwargs.update(dict(init_phase=0.3,
                              eta=0.6))
        D = DirichletReduced(**wg_kwargs)
        delta_min, delta_max = -0.65, 1.25

    eps, delta = D.get_cycle_parameters()
    limits = (eps_min, eps_max, eps_N,
              delta_min, delta_max, delta_N)
    x, y, z = D.sample_H(*limits)
    z_diff = z[...,0] - z[...,1]
    Z0 = np.sqrt(z_diff.real**2 + (z_diff.imag)**2)

    f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True,
                                 figsize=(6.2, 5./3.), dpi=220)
    if pos_dep:
        vmax_real = 2.0
        vmax_imag = 8.0
    else:
        vmax_real = 2.0
        vmax_imag = 1.0
    Z1 = np.abs(z_diff.real)
    p1 = ax1.imshow(Z1, cmap=cmap, aspect='auto', origin='lower',
                    extent=[y.min(), y.max(), x.min(), x.max()],
                    vmin=0.0, vmax=vmax_real)
    Z2 = np.abs(z_diff.imag)
    p2 = ax2.imshow(Z2, cmap=cmap, aspect='auto', origin='lower',
                    extent=[y.min(), y.max(), x.min(), x.max()],
                    vmin=0.0, vmax=vmax_imag)

    # mlab.surf(z[..., 0].imag)
    # mlab.surf(z[..., 1].imag)
    # mlab.show()

    for (p, ax) in zip((p1, p2), (ax1, ax2)):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        if ax == ax1:
            loc = MultipleLocator(vmax_real/2.)
        else:
            loc = MultipleLocator(vmax_imag/2.)
        cb = plt.colorbar(p, cax=cax, ticks=loc, format="%.1f")
        cb.ax.tick_params(labelsize=10)
        cb.solids.set_edgecolor('face')

    idx = np.argmin(Z0)
    x_EP, y_EP = [u.ravel()[idx] for u in (x,y)]
    print "x_EP", x_EP
    print "y_EP", y_EP

    dot_kwargs = dict(ms=6.0, mec='w', clip_on=False, zorder=10)
    for ax in (ax1, ax2):
        ax.plot(delta, eps, "w-", zorder=10)
        ax.plot(delta[0], eps[0], "wo", **dot_kwargs)
        ax.plot(delta[-1], eps[-1], "wo", **dot_kwargs)
        ax.set_ylim(eps_min, eps_max)
        ax.set_xlim(delta_min, delta_max)
        if pos_dep:
            ax.scatter(y_EP, x_EP, color="w")
            ax.xaxis.set_major_locator(MultipleLocator(1.00))
            ax.annotate('EP', (-0.35, 0.005), textcoords='data',
                        weight='bold', size=14, color='white')
        else:
            ax.scatter(D.y_EP, D.x_EP, color="w")
            ax.xaxis.set_major_locator(MultipleLocator(0.50))
            ax.annotate('EP', (0.1, 0.04), textcoords='data',
                        weight='bold', size=14, color='white')

        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        # ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    if pos_dep:
        datay = [y_EP, -0.278, -0.234, -0.188, -0.140, -0.072, -0.048, -0.024, -0.007, 0.000, 0.0]
        datax = [x_EP,  0.033,  0.034,  0.034,  0.033,  0.028,  0.025,  0.021,  0.015, 0.008, -0.1]
        ax1.plot(datay, datax, "w--", lw=0.5, dashes=[4, 3])
        datay = [y_EP, -0.352, -0.416, -0.506, -0.596, -0.650, -0.684, -0.736, -0.791, -0.827, -0.901, -0.972, -1.058, -1.145]
        datax = [x_EP,  0.030,  0.026,  0.018,  0.011,  0.007,  0.005,  0.002,  0.000,  0.000,  0.000,  0.003,  0.008,  0.015]
        ax2.plot(datay, datax, "w--", lw=0.5, dashes=[4, 3])
    else:
        ax1.plot([0, D.y_EP], [-10, D.x_EP], "w--", lw=0.5, dashes=[4, 3])
        ax2.plot([0, D.y_EP], [10, D.x_EP], "w--", lw=0.5, dashes=[4, 3])

    ax1.set_ylabel(r'Amplitude $\varepsilon$')
    f.text(0.45, 0.0, r'Detuning $\delta$', va='center')

    for (ax, text) in zip((ax1, ax2), ('a', 'b')):
        ax.annotate(text, (-0.90, 0.08), textcoords='data',
                    weight='bold', size=14, color='white')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    if show:
        plt.show()
    else:
        if pos_dep:
            plt.savefig("branch_cut_pos_dep.pdf", bbox_inches='tight')
        else:
            plt.savefig("branch_cut_constant.pdf", bbox_inches='tight')

if __name__ == '__main__':
    argh.dispatch_command(plot_spectrum)
