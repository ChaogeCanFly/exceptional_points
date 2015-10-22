#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import splprep, splev

import argh

from ep.waveguide import DirichletReduced
from ep.waveguide import DirichletPositionDependentLossReduced
from ep.plot import get_colors, get_defaults

colors, cmap, _ = get_colors()
get_defaults()


def plot_spectrum(fig=None, ax1=None, ax2=None, pos_dep=False,
                  eps_min=None, eps_max=None, eps_N=None, delta_N=None,
                  interpolate=False):

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
    else:
        wg_kwargs.update(dict(init_phase=0.3,
                              eta=0.6))
        D = DirichletReduced(**wg_kwargs)
        delta_min, delta_max = -0.65, 1.25

    eps, delta = D.get_cycle_parameters()
    limits = (eps_min, eps_max, eps_N,
              delta_min, delta_max, delta_N)
    x, y, z = D.sample_H(*limits)
    z_diff = z[..., 0] - z[..., 1]
    Z0 = np.sqrt(z_diff.real**2 + (z_diff.imag)**2)

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
    x_EP, y_EP = [u.ravel()[idx] for u in (x, y)]
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
            # eps^1 scaling
            # ax.annotate('EP', (-0.35, 0.005), textcoords='data',
            #             weight='bold', size=14, color='white')
            # eps^2 scaling
            ax.annotate('EP', (-0.325, 0.0175), textcoords='data',
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

    if pos_dep:
        # eps^1 scaling
        # datay = [y_EP, -0.278, -0.234, -0.188, -0.140, -0.072, -0.048, -0.024, -0.007, 0.000, 0.0]
        # datax = [x_EP,  0.033,  0.034,  0.034,  0.033,  0.028,  0.025,  0.021,  0.015, 0.008, -0.1]
        # eps^2 scaling
        data = np.asarray([
            [-0.242, 0.046],
            [-0.175, 0.049],
            [-0.121, 0.049],
            [-0.070, 0.047],
            [-0.035, 0.041],
            [-0.016, 0.032],
            [-0.010, 0.021],
            [-0.003, 0.001],
            [0.000, -0.009],
            ])
        datay, datax = data.T

        if interpolate:
            tck, _ = splprep([datay, datax], s=0.0001, k=3)
            datay, datax = splev(np.linspace(0, 1, 10), tck)
            np.savetxt("interpolate_coordinates.dat", np.vstack([datay, datax]).T,
                       fmt="[%.3f,%.3f],", delimiter=",", header="[", footer="]",
                       comments='')
        ax1.plot(datay, datax, "w--", ms=1, lw=0.5, dashes=[4, 3])
        # eps^1 scaling
        # datay = [y_EP, -0.352, -0.416, -0.506, -0.596, -0.650, -0.684, -0.736, -0.791, -0.827, -0.901, -0.972, -1.058, -1.145]
        # datax = [x_EP,  0.030,  0.026,  0.018,  0.011,  0.007,  0.005,  0.002,  0.000,  0.000,  0.000,  0.003,  0.008,  0.015]
        # eps^2 scaling
        data = np.asarray([
            [-0.240, 0.046],
            [-0.390, 0.032],
            [-0.513, 0.021],
            [-0.603, 0.013],
            [-0.675, 0.007],
            [-0.766, 0.002],
            [-0.853, -0.001],
            [-0.954, 0.003],
            [-1.024, 0.006],
            [-1.096, 0.011],
            ]).T
        datay, datax = data
        ax2.plot(datay, datax, "w--", ms=1, lw=0.5, dashes=[4, 3])
    else:
        ax1.plot([0, D.y_EP], [-10, D.x_EP], "w--", lw=0.5, dashes=[4, 3])
        ax2.plot([0, D.y_EP], [10, D.x_EP], "w--", lw=0.5, dashes=[4, 3])

    ax1.set_ylabel(r'Amplitude $\varepsilon$')
    fig.text(0.45, 0.0, r'Detuning $\delta$', va='center')
    fig.text(-0.00, 0.92, 'a', weight='bold', size=14, color='black')
    fig.text(-0.00, 0.45, 'b', weight='bold', size=14, color='black')

    # for (ax, text) in zip((ax1, ax2), ('a', 'b')):
    #     ax.annotate(text, (-0.90, 0.08), textcoords='data',
    #                 weight='bold', size=14, color='white')

    return ax1, ax2


def on_pick(event, event_coordinates, ax):
    """Record (x, y) coordinates at each click and print to file."""
    xmouse, ymouse = event.xdata, event.ydata

    if event.button == 1:
        print "x, y:", xmouse, ymouse
        event_coordinates.append([xmouse, ymouse])
        x, y = np.asarray(event_coordinates).T
        ax.scatter(x, y, s=1e1, c="k", edgecolors=None)
    elif event.button == 3:
        x, y = np.asarray(event_coordinates).T
        x_close = np.isclose(x, xmouse, rtol=1e-2)
        y_close = np.isclose(y, ymouse, rtol=1e-2)
        try:
            idx = np.where(x_close & y_close)[0][0]
            ax.scatter(x[idx], y[idx], s=1e1, c="red", marker="x")
            del event_coordinates[idx]
        except:
            print "No point found near ({}, {}).".format(xmouse, ymouse)

    ax.get_figure().canvas.draw()


def build_composite_plot(eps_min=-0.01, eps_max=0.11, eps_N=101, delta_N=101,
                         show=False, interactive=False, interpolate=False):
    plot_kwargs = locals()
    plot_kwargs.pop('show')
    plot_kwargs.pop('interactive')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                               sharex=False, sharey=True,
                                               figsize=(6.2, 10./3.), dpi=220)
    ax1, ax2 = plot_spectrum(fig=f, ax1=ax1, ax2=ax2,
                             pos_dep=False, **plot_kwargs)
    ax3, ax4 = plot_spectrum(fig=f, ax1=ax3, ax2=ax4,
                             pos_dep=True, **plot_kwargs)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    if show:
        if interactive:
            event_coordinates = []
            on_pick_lambda = lambda e: on_pick(e, event_coordinates, ax4)
            f.canvas.mpl_connect('button_press_event', on_pick_lambda)
        plt.show()
    else:
        plt.savefig("branch_cuts.pdf", bbox_inches='tight')

    if interactive:
        np.savetxt("interactive_coordinates.dat", np.asarray(event_coordinates),
                   fmt="[%.3f,%.3f],", delimiter=",", header="[", footer="]",
                   comments='')

if __name__ == '__main__':
    argh.dispatch_command(build_composite_plot)
