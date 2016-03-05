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


def main(W=0.05, L=25, phase=None, plot=False, save_plot=False):
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

    def xi(eps, delta, x=x):
        return eps*np.sin((WG_exp.kr + delta)*x)

    f = plt.figure(figsize=(6.3, 3), dpi=220)
    ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1)

    f.text(0.0, 0.95, 'a', weight='bold', size=12)
    f.text(0.0, 0.45, 'b', weight='bold', size=12)

    configuration_labels = ("I", "II", "III", "IV", "V")

    # experimental WG
    ax1.plot(x, -xi(eps, delta) + W, "k-", lw=0.75)
    ax1.plot(x, -xi(eps, delta), "k-", lw=0.75)
    ax1.plot(x, y_absorber - xi(eps, delta), ls="-", lw=3, color=c[1])
    ax1.set_xlabel(r"Spatial coordinate $x$ (m)") #, labelpad=0.0)
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
    ax2.plot(delta_linearized*W, eps_linearized/W, "k-", lw=0.75)
    ax2.plot(delta_linearized[absorber_cutoff]*W,
             eps_linearized[absorber_cutoff]/W, ls="-", lw=3, color=c[1])
    ax2.set_xlabel(r"$\delta\cdot W$") #, labelpad=-5.0)
    ax2.set_ylabel(r"$\sigma/W$") #, labelpad=2.0)
    ax2.set_xlim(-3.1, 2)
    ax2.set_ylim(0.0, 0.18)
    ax2.set_yticks(np.linspace(0.0, 0.18, 4))

    ax22 = ax2.twiny()
    ax22.set_xlim(-3.1, 2)
    ax22.set_xticks(snapshots_delta_values)

    for ax in (ax1, ax11, ax2, ax22):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.tick_params(axis='both', which='minor', bottom='off',
                       left='off', right='off', top='off')
        ax.tick_params(axis='both', which='major', bottom='on',
                      left='on', right='off', top='off')

    for ax in (ax11, ax22):
        ax.grid(True, lw=1.)
        ax.set_xticklabels(configuration_labels)
        ax.tick_params(axis='both', which='major', bottom='off',
                        left='on', right='off', top='off')
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(-4.0)

    if plot:
        plt.tight_layout(pad=1.25)
        f.subplots_adjust(bottom=0.15, left=0.1, top=0.9, right=0.95)
        if save_plot:
            plt.savefig("waveguide.pdf")
        else:
            plt.show()


if __name__ == '__main__':
    argh.dispatch_command(main)
