#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.plot import get_colors, get_defaults
from ep.waveguide import DirichletPositionDependentLossReduced, DirichletReduced

get_defaults()
colors, _, _ = get_colors()


def get_toymodel_data():
    wg_kwargs = dict(N=2.6,
            L=25.0,
            W=1.0,
            x_R0=0.16,
            y_R0=2.5,
            init_phase=-0.55,
            switch_losses_on_off=True,
            eta=1.0,
            eta0=1.0,
            loop_type='Bell')
    D = DirichletPositionDependentLossReduced(**wg_kwargs)
    _, b1, b2 = D.solve_ODE()
    eps, delta = D.get_cycle_parameters()
    v1, v2 = [D.eVecs_r[:, :, m] for m in (0, 1)]

    return eps, delta, v1, v2

# def get_toymodel_data():
#     wg_kwargs = dict(N=2.6,
#             L=25.0,
#             W=1.0,
#             x_R0=0.16,
#             y_R0=2.5,
#             init_phase=-0.55,
#             switch_losses_on_off=True,
#             eta=2.5,
#             loop_type='Bell')
#     D = DirichletReduced(**wg_kwargs)
#     _, b1, b2 = D.solve_ODE()
#     eps, delta = D.get_cycle_parameters()
#     v1, v2 = [D.eVecs_r[:, :, m] for m in (0, 1)]
#
#     return eps, delta, v1, v2


@argh.arg("--exp-config", type=int, nargs='+')
@argh.arg("--exp-file", type=str, nargs='+')
def plot_coefficients(save=False, evecs_file="evecs_t.dat", exp_file=None,
                      exp_config=None, lw=1.0, plot_evals=False):
    """docstring for plot_coefficients"""

    c = (-1.65, -1.1, -0.55, 0.0, 0.55)

    (ev1_abs, ev1_phi, v1r, v1i, v2r, v2i,
            ev2_abs, ev2_phi, c1r, c1i, c2r, c2i) = np.loadtxt(evecs_file).T
    v1 = v1r + 1j*v1i
    v2 = v2r + 1j*v2i
    c1 = c1r + 1j*c1i
    c2 = c2r + 1j*c2i

    # plot
    dpi = 220
    if plot_evals:
        f, (ax2, ax1) = plt.subplots(ncols=2, figsize=(6.3, 2), dpi=dpi)
    else:
        f, ax1 = plt.subplots(ncols=1, figsize=(6.3/1.75, 2.5), dpi=dpi)

    if plot_evals:
        f.text(0.0, 0.9, 'a', weight='bold', size=12)
        f.text(0.5, 0.9, 'b', weight='bold', size=12)

    exp_data = []
    exp_data_evals = []
    for expn, (exp_c, exp_f) in enumerate(zip(exp_config, exp_file)):
        (ev1_abs_exp, ev1_phi_exp, v1r, v1i, v2r, v2i,
                ev2_abs_exp, ev2_phi_exp, c1r, c1i, c2r, c2i) = np.loadtxt(exp_f).T
        v1_exp = v1r + 1j*v1i
        v2_exp = v2r + 1j*v2i
        c1_exp = c1r + 1j*c1i
        c2_exp = c2r + 1j*c2i
        exp_data.append([c[exp_c], v1_exp, v2_exp, c1_exp, c2_exp])
        exp_data_evals.append([ev1_abs_exp, ev1_phi_exp, ev2_abs_exp, ev2_phi_exp])

    c_exp, v1_exp, v2_exp, c1_exp, c2_exp = np.asarray(exp_data).T
    ev1_abs_exp, ev1_phi_exp, ev2_abs_exp, ev2_phi_exp = np.asarray(exp_data_evals).T

    ax1.set_ylabel(r"$\arctan\vert c^{(n)}_1/c^{(n)}_2\vert$")
    ax1.set_xlabel(r"$\delta\cdot W$")
    marker1 = "s"
    marker2 = "^"
    marker3 = "D"
    marker4 = "v"
    ms_1 = 5.5
    ms_2 = 5.5
    ms_exp_1 = 4.5
    ms_exp_2 = 5.5
    if 0:
        marker1 = "o"
        marker2 = "o"
        marker3 = "v"
        marker4 = "v"
        ms_1 = 5.0
        ms_2 = 5.0
        ms_exp_1 = 5.0
        ms_exp_2 = 5.0
    ms_end_1 = ms_exp_1
    ms_end_2 = ms_exp_2
    ax1.plot(c, np.arctan(abs(v1/v2)), marker1, color=colors[0], clip_on=False, mfc='none', mec=colors[0], lw=lw, ms=ms_1)
    ax1.plot(c, np.arctan(abs(c1/c2)), marker2, color=colors[1], clip_on=False, mfc='none', mec=colors[1], lw=lw, ms=ms_2)
    ax1.plot(c_exp, np.arctan(abs(v1_exp/v2_exp)), marker3, color=colors[0], mfc="none", mec=colors[0], clip_on=False, lw=lw, ms=ms_exp_1)
    ax1.plot(c_exp, np.arctan(abs(c1_exp/c2_exp)), marker4, color=colors[1], mfc="none", mec=colors[1], clip_on=False, lw=lw, ms=ms_exp_2)

    eps_tm, delta_tm, v1_tm, v2_tm = get_toymodel_data()
    ax1.plot(delta_tm, np.arctan(abs(v1_tm[:, 0]/v1_tm[:, 1])), ls="--", color=colors[0], lw=lw)
    ax1.plot(delta_tm, np.arctan(abs(v2_tm[:, 0]/v2_tm[:, 1])), ls="--", color=colors[1], lw=lw)

    c_exp_extended = np.concatenate([[delta_tm[0]], [delta_tm[-1]]])
    arctan_v_exp_extended = np.concatenate([[0], [np.pi/2.]])
    arctan_c_exp_extended = np.concatenate([[np.pi/2.], [0.0]])
    for color_n, arctan in enumerate([arctan_v_exp_extended, arctan_c_exp_extended]):
        for idx in (0, -1):
            if color_n == 0:
                marker = marker3
                marker = marker1
                ms_end = ms_end_1
            else:
                marker = marker4
                marker = marker2
                ms_end = ms_end_2
            ax1.plot(c_exp_extended[idx], arctan[idx], marker,
                     color=colors[color_n], mec=colors[color_n], mfc='none',
                     clip_on=False, lw=lw, ms=ms_end)

    ax1.set_ylim(-0.1, np.pi/2)
    ax1.set_yticks([0, np.pi/4, np.pi/2])
    ax1.set_yticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])

    ev_list = [ev1_abs*np.exp(1j*ev1_phi),
               ev2_abs*np.exp(1j*ev2_phi),
               ev1_abs_exp*np.exp(1j*ev1_phi_exp),
               ev2_abs_exp*np.exp(1j*ev2_phi_exp)]
    ev1, ev2, ev1_exp, ev2_exp = ev_list
    # WG_len = [0.51576837377840601,
    #           2.*0.49262250087499387,
    #           4.*0.39514141195540475,
    #           4.*0.4792450151610923,
    #           4.*0.53112071257820737]
    # ev1, ev2, ev1_exp, ev2_exp = [ e**(1./w) for e, w in zip(ev_list, WG_len)]

    if plot_evals:
        # absolute value
        ax2.set_ylabel(r"$\vert \tau_n \vert$")
        ax2.semilogy(c, np.abs(ev1), marker="o", ls="-", color=colors[0], clip_on=False, mec='none', lw=lw)
        ax2.semilogy(c, np.abs(ev2), marker="o", ls="-", color=colors[1], clip_on=False, mec='none', lw=lw)
        ax2.semilogy(c_exp, np.abs(ev1_exp), marker="v", ls=":", color=colors[0], mfc="w", mec=colors[0], clip_on=False, lw=lw)
        ax2.semilogy(c_exp, np.abs(ev2_exp), marker="v", ls=":", color=colors[1], mfc="w", mec=colors[1], clip_on=False, lw=lw)
        ax2.set_xlabel(r"$\delta\cdot W$")

    print "eigenvalue ratios |tau_1|/|tau_2|"
    print "num:", np.abs(ev1/ev2)
    print "exp:", np.abs(ev1_exp/ev2_exp)

    # phase
    # ax2.set_ylabel(r"$\operatorname{Arg} \tau_n$")
    # ax2.plot(c, np.angle(ev1), marker="o", ls="-", color=colors[0], clip_on=False, mec='none', lw=lw)
    # ax2.plot(c, np.angle(ev2), marker="o", ls="-", color=colors[1], clip_on=False, mec='none', lw=lw)
    # ax2.plot(c_exp, np.angle(ev1_exp), marker="v", ls=":", color=colors[0], clip_on=False, mec=colors[0], mfc="w", lw=lw)
    # ax2.plot(c_exp, np.angle(ev2_exp), marker="v", ls=":", color=colors[1], clip_on=False, mec=colors[1], mfc="w", lw=lw)

    if plot_evals:
        axes_list = (ax1, ax2)
    else:
        axes_list = (ax1, )

    for ax in axes_list:
        ax.set_xlim(-3.2, 2.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', bottom='on',
                left='on', right='off', top='off')
        ax.tick_params(axis='both', which='minor', bottom='off',
                left='off', right='off', top='off')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')

    plt.tight_layout(pad=1.15)

    if not save:
        plt.show()
    else:
        plt.savefig("arctan.pdf")


if __name__ == '__main__':
    argh.dispatch_command(plot_coefficients)
