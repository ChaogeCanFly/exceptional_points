#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import interp1d

import argh

from ep.waveguide import DirichletReduced
from ep.plot import get_colors


c, _, _ = get_colors()


@argh.arg("-p", "--plot-phase", type=float)
@argh.arg("--threshold-left", type=float)
@argh.arg("--threshold-right", type=float)
def main(W=0.05, L=25, config=1, plot_phase=None, plot=False, save_plot=False, threshold_left=None, threshold_right=None):
    """docstring for main"""

    L = L*W
    snapshots_x_values = [7*W, 9.75*W, L/2, 15.25*W, 18*W]

    exp_setup = dict(N=2.6,
                     L=L,
                     W=W,
                     x_R0=0.16*W,
                     y_R0=1.25/W,
                     init_phase=-1.8/W,
                     loop_type='Bell',
                     tN=500)
    WG_exp = DirichletReduced(**exp_setup)

    effective_setup = exp_setup.copy()
    effective_setup.update(dict(y_R0=2.*exp_setup['y_R0'],
                                init_phase=exp_setup['init_phase'] + exp_setup['y_R0']))
    WG_eff = DirichletReduced(**effective_setup)

    # show eps, delta values at start/end of absorber
    snapshots_delta_values = []
    for n, s in enumerate(snapshots_x_values):
        s_eps_delta = WG_eff.get_cycle_parameters(s)
        print "configuration {} at x={:.5f}: eps={:.5f} delta={: .5f}".format(n, s, s_eps_delta[0]/W, s_eps_delta[1]*W)
        snapshots_delta_values.append(s_eps_delta[-1]*W)

        if config == n:
            eps_reduced_model, delta_reduced_model = s_eps_delta
            x0 = s

    if plot_phase is None and threshold_left is None and threshold_right is None:
        if config == 0:
            plot_phase = 1.4
            threshold_left = 0.35
            threshold_left_absorber = 0.465
        elif config == 1:
            plot_phase = -np.pi
            threshold_left = 0.1925
            threshold_left_absorber = 0.225
        elif config == 2:
            plot_phase = np.pi
            threshold_left = 0.31
            threshold_left_absorber = 0.32
        elif config == 3:
            plot_phase = +1.9
            threshold_left = 0.507
            threshold_left_absorber = 0.522
        elif config == 4:
            plot_phase = -1.0
            threshold_left = 0.7375
            threshold_left_absorber = 0.705

    plot_phase = -2.*WG_exp.y_R0*x0**2/L

    eps, delta = WG_exp.get_cycle_parameters()
    x = WG_exp.t

    f_absorber = interp1d(*np.loadtxt("peaks_interactive.dat", unpack=True))
    y_absorber = 1.*x
    absorber_cutoff = (x > 7.*W) & (x < 18.*W)
    y_absorber[absorber_cutoff] = f_absorber(x[absorber_cutoff])
    y_absorber[x < 7*W] = np.nan
    y_absorber[x > 18*W] = np.nan

    def xi(eps, delta, plot_phase=0.0, x=x):
        return eps*np.sin((WG_exp.kr + delta)*x + plot_phase)

    f, (ax, ax3, ax4) = plt.subplots(nrows=3, figsize=(10., 6), dpi=120)

    # experimental WG
    ax.plot(x, -xi(eps, delta) + W, "k-")
    ax.plot(x, -xi(eps, delta), "k-")
    ax.plot(x, y_absorber - xi(eps, delta), ls="-", lw=3, color=c[1])
    ax.set_xlabel(r"$x$", labelpad=-5.0)
    ax.set_ylabel(r"$y$", labelpad=2.0)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.01, 0.06)

    ax2 = ax.twiny()
    ax2.set_xlim(0, L)
    ax2.set_xticks(snapshots_x_values)
    ax2.set_xticklabels([str(t) for t in snapshots_x_values])
    for tick in ax2.get_xaxis().get_major_ticks():
        tick.set_pad(0.0)
    ax2.grid(True, lw=1.)

    # linearized 2x2 parameter path
    eps_linearized, delta_linearized = WG_eff.get_cycle_parameters()
    ax3.plot(delta_linearized*W, eps_linearized/W, "k-", lw=0.75)
    ax3.plot(delta_linearized[absorber_cutoff]*W,
             eps_linearized[absorber_cutoff]/W, ls="-", lw=3, color=c[1])
    ax3.set_xlabel(r"$\delta\cdot W$", labelpad=-5.0)
    ax3.set_ylabel(r"$\sigma/W$", labelpad=2.0)
    ax3.set_xlim(-3.1, 2)

    ax33 = ax3.twiny()
    ax33.set_xlim(-3.1, 2)
    ax33.set_xticks(snapshots_delta_values)
    ax33.set_xticklabels(["{:.3f}".format(t) for t in snapshots_delta_values])
    for tick in ax33.get_xaxis().get_major_ticks():
        tick.set_pad(0.0)
    ax33.grid(True, lw=1.)

    snapshots_y_values = [0.095, 0.1416, 0.16]
    ax333 = ax3.twinx()
    ax333.set_ylim(0, 0.18)
    ax333.set_yticks(snapshots_y_values)
    ax333.set_yticklabels([str(t) for t in snapshots_y_values])
    for tick in ax33.get_yaxis().get_major_ticks():
        tick.set_pad(0.0)
    ax333.grid(True, lw=1.)

    # comparison periodic system and experiment
    ax4.plot(x, -xi(eps, delta) + W, "k-", lw=0.25)
    ax4.plot(x, -xi(eps, delta), "k-", lw=0.25)
    ax4.plot(x, y_absorber - xi(eps, delta), ls="-", lw=1, color=c[1])
    xi_periodic = xi(eps_reduced_model, delta_reduced_model, plot_phase)
    ax4.plot(x, -xi_periodic, "k-")
    ax4.plot(x, W - xi_periodic, "k-")
    ax4.set_xlim(0, L)
    ax4.set_ylim(-0.01, 0.06)
    ax4.set_xlabel(r"$x$", labelpad=-5.0)
    ax4.set_ylabel(r"$y$", labelpad=2.0)

    ax5 = ax4.twiny()
    ax5.set_xlim(0, L)
    ax5.set_xticks(snapshots_x_values)
    ax5.set_xticklabels([str(t) for t in snapshots_x_values])
    for tick in ax5.get_xaxis().get_major_ticks():
        tick.set_pad(0.0)
    ax5.grid(True, lw=1.)

    # extract a half period of the absorber
    wavelength = 2.*np.pi/(WG_eff.kr + delta_reduced_model)
    dx = wavelength/4.
    piece_mask = (x > x0 - dx) & (x < x0 + dx)
    a = y_absorber[piece_mask]
    ax5.plot(x[piece_mask], a - xi_periodic[piece_mask], "y-")

    for axis in (ax, ax3, ax4):
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax4.set_ylim(-0.01, 0.06)

    periodic_absorber = np.concatenate([a, a[::-1]]*4 + [a])
    elements = len(periodic_absorber)/len(a)

    if config == 0:
        piece_mask = (x > x0) & (x < x0 + dx)
        a = y_absorber[piece_mask]
        periodic_absorber = np.concatenate(4*[W - a[::-1], W - a, a[::-1], a])
        elements = len(periodic_absorber)/len(a)/2.

    if config == 4:
        piece_mask = (x > x0 - dx) & (x < x0)
        a = y_absorber[piece_mask]
        periodic_absorber = np.concatenate(4*[W - a[::-1], W - a, a[::-1], a])
        elements = len(periodic_absorber)/len(a)/2.

    x_elements = np.linspace(x0 - elements*dx,
                             x0 + elements*dx, len(periodic_absorber))

    # start at maximum of boundary oscillation -> different for each configuration
    maximum_mask = (x_elements > threshold_left) & (x_elements < threshold_left + 4*wavelength)
    maximum_mask_absorber = (x_elements > threshold_left_absorber) & (x_elements < threshold_left_absorber + 4*wavelength)

    x_file = x_elements
    y_file = periodic_absorber
    xi_file = xi(eps_reduced_model, delta_reduced_model, x=x_file)

    m = maximum_mask_absorber
    ax5.set_xticks(snapshots_x_values + [x0 - dx, x0 + dx])
    ax5.plot(x_file[m], y_file[m] -
             xi(eps_reduced_model, delta_reduced_model, plot_phase=plot_phase, x=x_file)[m], "r-")

    if plot:
        plt.tight_layout()
        f.subplots_adjust(bottom=0.1, left=0.09, top=0.9, right=0.9)
        if save_plot:
            plt.savefig("summary_config_{}".format(config))
        else:
            plt.show()

    # save file: x in [0, 4*lambda], y in [0, 1]
    print
    print "kr", WG_exp.kr
    print "delta_reduced_model", delta_reduced_model
    print "Omega", (WG_exp.kr + delta_reduced_model)
    print
    print "wavelength", wavelength
    print "4*wavelength", 4*wavelength
    print "wavelength/W", wavelength/W
    print "4*wavelength/W", 4*wavelength/W
    print

    x_file = x_file[maximum_mask]
    y_file = y_file[maximum_mask_absorber]
    xi_file = xi_file[maximum_mask]

    print "len(x_file)", len(x_file)
    print "len(y_file)", len(y_file)
    print "len(xi_file)", len(xi_file)
    if len(x_file) > len(y_file):
        x_file = x_file[:len(y_file)]
        xi_file = xi_file[:len(y_file)]
    else:
        y_file = y_file[:len(x_file)]
    print "len(x_file)", len(x_file)
    print "len(y_file)", len(y_file)
    print "len(xi_file)", len(xi_file)

    np.savetxt("periodic_configuration_{}.dat".format(config),
               # zip((x_file - x_file[0])/(x_file[-1] - x_file[0])*4*wavelength/W,
               zip((x_file - x_file[0])/W,
                   y_file/W,
                   xi_file),
               header="x, y_absorber (relative coordinates), xi(x) (boundary modulation)")
    np.savetxt("experimental_configuration_{}.dat".format(config),
               # zip((x_file - x_file[0])/(x_file[-1] - x_file[0])*4*wavelength,
               zip(x_file - x_file[0],
                   2*W - (y_file + xi_file + eps_reduced_model),
                   2*W - (xi_file + eps_reduced_model),
                   2*W - (xi_file + eps_reduced_model + W)),
               header="x, y_absorber (absolute coordinates), xi_lower xi_upper")


if __name__ == '__main__':
    argh.dispatch_command(main)
