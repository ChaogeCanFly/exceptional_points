#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import interp1d

import argh

from ep.waveguide import DirichletReduced
from ep.plot import get_colors, get_defaults


get_defaults()
c, _, _ = get_colors()


def main(W=0.05, L=25, phase=None, plot=False, save_plot=False, lw=0.75):
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

    eps, delta = WG_exp.get_cycle_parameters()
    x = WG_exp.t

    f_absorber = interp1d(*np.loadtxt("peaks_interactive_RAP.dat", unpack=True))
    y_absorber = 1.*x
    absorber_cutoff = (x > 7.*W) & (x < 18.*W)
    y_absorber[absorber_cutoff] = f_absorber(x[absorber_cutoff])
    y_absorber[x < 7*W] = np.nan
    y_absorber[x > 18*W] = np.nan

    def xi(eps, delta, x=x, plot_phase=0.0):
        return eps*np.sin((WG_exp.kr + delta)*x + plot_phase)

    f = plt.figure(figsize=(6.3, 4.0), dpi=220)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    ax3 = plt.subplot2grid((3, 1), (1, 0), rowspan=1)

    f.text(0.0, 0.95, 'a', weight='bold', size=12)
    f.text(0.0, 0.61, 'b', weight='bold', size=12)
    f.text(0.0, 0.30, 'c', weight='bold', size=12)

    configuration_labels = ("I", "II", "III", "IV", "V")

    # experimental WG
    ax1.plot(x, -xi(eps, delta) + W, "k-", lw=lw)
    ax1.plot(x, -xi(eps, delta), "k-", lw=lw)
    ax1.plot(x, y_absorber - xi(eps, delta), ls="-", lw=3, color=c[1])
    ax1.set_xlabel(r"Spatial coordinate $x$ (m)", labelpad=1.5)
    ax1.set_ylabel(r"$y$") #, labelpad=2.0)
    ax1.set_xlim(0, L)
    ax1.set_ylim(-0.01, 0.06)
    ax1.set_xticks([0, 7*W, 12.5*W, 18*W, L])
    ax1.set_xticklabels([r"$0$", r"$7W$", r"$12.5W$", r"$18W$", r"$25W$"])
    ax1.set_yticks([0.0, 0.05])
    ax1.set_yticklabels([r"$0$", r"$W$"])

    ax11 = ax1.twiny()
    ax11.set_xlim(0, L)
    ax11.set_xticks(snapshots_x_values)

    # linearized 2x2 parameter path
    eps_linearized, delta_linearized = WG_eff.get_cycle_parameters()
    ax2.plot(delta_linearized*W, eps_linearized/W, "k-", lw=lw)
    ax2.plot(delta_linearized[absorber_cutoff]*W,
             eps_linearized[absorber_cutoff]/W, ls="-", lw=3, color=c[1])
    ax2.set_xlabel(r"$\delta\cdot W$", labelpad=1.5)
    ax2.set_ylabel(r"$\sigma/W$") #, labelpad=2.0)
    ax2.set_xlim(-3.1, 2)
    ax2.set_ylim(0.0, 0.18)
    ax2.set_yticks(np.linspace(0.0, 0.18, 4))

    ax22 = ax2.twiny()
    ax22.set_xlim(-3.1, 2)
    ax22.set_xticks(snapshots_delta_values)

    # plot snapshot III
    x0 = snapshots_x_values[2]
    eps_reduced_model, delta_reduced_model = WG_eff.get_cycle_parameters(x0)
    plot_phase = -2.*WG_exp.y_R0*x0**2/L
    xi_periodic = xi(eps_reduced_model, delta_reduced_model, plot_phase=plot_phase)
    wavelength = 2.*np.pi/(WG_eff.kr + delta_reduced_model)
    dx = wavelength/4.
    x = WG_eff.t
    a = y_absorber[(x > x0 - dx) & (x < x0 + dx)]
    periodic_absorber = np.concatenate([a, a[::-1]]*4 + [a])
    elements = len(periodic_absorber)/len(a)
    x_elements = np.linspace(x0 - elements*dx,
                             x0 + elements*dx, len(periodic_absorber))
    plot_mask = (x_elements > 0.32) & (x_elements < 0.32 + 4*wavelength)
    x_file = x_elements[plot_mask]
    y_file = periodic_absorber[plot_mask]
    xi_file = xi(eps_reduced_model, delta_reduced_model, x=x_file, plot_phase=plot_phase)

    ax3.plot(x, -xi(eps, delta) + W, "-", color=c[-1], lw=lw)
    ax3.plot(x, -xi(eps, delta), "-", color=c[-1], lw=lw)
    ax3.plot(x, y_absorber - xi(eps, delta), ls="-", lw=3., color=c[-1])
    ax3.set_xlabel(r"Spatial coordinate $x$ (m)", labelpad=1.5)
    ax3.set_ylabel(r"$y$")
    ax3.set_xlim(0, L)
    ax3.set_ylim(-0.01, 0.06)
    ax3.set_xticks([0, 7*W, 12.5*W, 18*W, L])
    ax3.set_xticklabels([r"$0$", r"$7W$", r"$12.5W$", r"$18W$", r"$25W$"])
    ax3.set_yticks([0.0, 0.05])
    ax3.set_yticklabels([r"$0$", r"$W$"])
    ax3.plot(x, -xi_periodic, "k-", lw=lw)
    ax3.plot(x, W - xi_periodic, "k-", lw=lw)
    ax3.plot(x_file, y_file -
             xi(eps_reduced_model, delta_reduced_model,
                plot_phase=plot_phase, x=x_file), "-", color=c[1], lw=3.)

    ax33 = ax3.twiny()
    ax33.set_xlim(0, L)
    ax33.set_xticks([12.5*W])
    ax33.set_xticklabels([r"III"])

    for ax in (ax1, ax11, ax2, ax22, ax3, ax33):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.tick_params(axis='both', which='minor', bottom='off',
                       left='off', right='off', top='off')
        ax.tick_params(axis='both', which='major', bottom='on',
                      left='on', right='off', top='off')

    for ax in (ax1, ax2, ax3):
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(1.0)

    for ax in (ax11, ax22, ax33):
        ax.grid(True, lw=1.)
        if ax != ax33:
            ax.set_xticklabels(configuration_labels)
        ax.tick_params(axis='both', which='major', bottom='off',
                        left='on', right='off', top='off')
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(-4.0)

    if plot:
        plt.tight_layout(pad=1.25)
        f.subplots_adjust(bottom=0.10, left=0.1, top=0.95, right=0.95)
        if save_plot:
            plt.savefig("waveguide.pdf")
        else:
            plt.show()


if __name__ == '__main__':
    argh.dispatch_command(main)
