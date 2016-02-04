#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import argh

from ep.waveguide import DirichletReduced
from ep.plot import get_colors


c, _, _ = get_colors()


def main(W=0.05, L=25, eps_config=0.16, delta_config=-0.55):
    """docstring for main"""

    L = L*W
    eps_rep, delta_rep = eps_config*W, delta_config/W

    exp_setup = dict(N=2.6,
                     L=L,
                     W=W,
                     x_R0=0.16*W,
                     y_R0=1.25/W,
                     init_phase=-1.8/W,
                     loop_type='Bell',
                     tN=1000)
    WG_exp = DirichletReduced(**exp_setup)

    effective_setup = exp_setup.copy()
    effective_setup.update(dict(x_R0=eps_rep,
                                y_R0=delta_rep,
                                init_phase=exp_setup['init_phase'] + delta_rep))
    WG_eff = DirichletReduced(**effective_setup)

    eps, delta = WG_exp.get_cycle_parameters()
    x = WG_exp.t

    # x_abs, y_abs = np.loadtxt("peaks_interactive.dat", unpack=True)
    # f_abs = interp1d(x_abs, y_abs)
    f_abs = interp1d(*np.loadtxt("peaks_interactive.dat", unpack=True))
    y_absorber = 1.*x
    absorber_cutoff = (x > 7.*W) & (x < 18.*W)
    y_absorber[absorber_cutoff] = f_abs(x[absorber_cutoff])
    y_absorber[x < 7*W] = np.nan
    y_absorber[x > 18*W] = np.nan

    def xi(eps, delta, phase=0.0, x=x):
        return eps*np.sin((WG_exp.kr + delta)*x + phase)

    f, (ax, ax3, ax4) = plt.subplots(nrows=3, figsize=(45, 5), dpi=200)

    # experimental WG
    ax.plot(x, -xi(eps, delta) + W, "k-")
    ax.plot(x, -xi(eps, delta), "k-")
    ax.plot(x, y_absorber - xi(eps, delta), ls="-", lw=3, color=c[1])
    ax.set_xlabel(r"$x$", labelpad=0.0)
    ax.set_ylabel(r"$y$", labelpad=0.0)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.01, 0.06)

    ax2 = ax.twiny()
    ax2.set_xlim(0, L)
    ax2.set_xticks([7*W, L/2, 18*W])
    ax2.set_xticklabels([str(t) for t in (7*W, L/2, 18*W)])
    ax2.grid(True, lw=1.)

    # linearized 2x2 parameter path
    eps2, delta2 = WG_eff.get_cycle_parameters()
    ax3.plot(delta2*W, eps2/W, "k-", lw=0.75)
    ax3.plot(delta2[absorber_cutoff]*W, eps2[absorber_cutoff]/W, ls="-", lw=3, color=c[1])
    ax3.set_xlabel(r"$\delta$", labelpad=0.0)
    ax3.set_ylabel(r"$\sigma$")

    # comparison periodic system and experiment
    phase = np.pi
    ax4.plot(x, -xi(eps, delta) + W, "k-", lw=0.25)
    ax4.plot(x, -xi(eps, delta), "k-", lw=0.25)
    ax4.plot(x, y_absorber - xi(eps, delta), ls="-", lw=1, color=c[1])
    ax4.plot(x, -xi(0.16*W, -0.55/W, phase), "k-")
    ax4.plot(x, W - xi(0.16*W, -0.55/W, phase), "k-")
    ax4.set_xlim(0, L)
    ax4.set_ylim(-0.01, 0.06)
    ax4.set_xlabel(r"$x$", labelpad=0.0)
    ax4.set_ylabel(r"$y$", labelpad=0.0)

    ax5 = ax4.twiny()
    ax5.set_xlim(0, L)
    ax5.set_xticks([7*W, L/2, 18*W])
    ax5.set_xticklabels([str(t) for t in (7*W, L/2, 18*W)])
    ax5.grid(True, lw=1.)

    # show eps, delta values at start/end of absorber
    s1 = WG_eff.get_cycle_parameters(7*W)
    s2 = WG_eff.get_cycle_parameters(18*W)
    # print "eps = ", s1[0]/W, "delta = ", s1[1]*W
    # print "eps = ", s2[0]/W, "delta = ", s2[1]*W

    # extract a half period of the absorber
    wavelength = 2.*np.pi/(WG_eff.kr + delta_rep)
    dx = wavelength/4
    piece_mask = (x > L/2. - dx) & (x < L/2. + dx)
    a = y_absorber[piece_mask]
    periodic_absorber = np.concatenate([a, a[::-1], a, a[::-1], a, a[::-1], a, a[::-1], a])
    elements = len(periodic_absorber)/len(a)

    x_rep = np.linspace(L/2. - elements*dx, L/2. + elements*dx, len(periodic_absorber))
    file_mask = (x_rep > 0.31) #& (x_rep < 0.85)
    ax5.plot(x_rep[file_mask], periodic_absorber[file_mask] + xi(eps_rep, delta_rep, x=x_rep)[file_mask], "r-")
    # ax5.plot(x_rep, periodic_absorber+ xi(eps_rep, delta_rep, x=x_rep), "r-")
    # ax5.plot(0.31, 0.025, "ro")
    # ax5.plot(0.85, 0.025, "ro")

    # save file
    np.savetxt("periodic_configuration_sigma_{}_delta_{}.dat".format(eps_rep, delta_rep),
               zip(x_rep[file_mask], periodic_absorber[file_mask], xi(eps_rep, delta_rep, x=x_rep[file_mask])),
               header="x, y_absorber (absolute coordinates), xi(x) (boundary modulation)")

    # backmatter
    plt.tight_layout()
    f.subplots_adjust(bottom=0.1, left=0.09, top=0.9)
    plt.show()


if __name__ == '__main__':
    argh.dispatch_command(main)
