#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.plot import get_colors
from ep.waveguide import DirichletPositionDependentLossReduced, DirichletReduced

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
def plot_coefficients(save=False, evecs_file="evecs_t.dat", exp_file=None, exp_config=None):
    """docstring for plot_coefficients"""

    c = (-1.65, -1.1, -0.55, 0.0, 0.55)

    (ev1_abs, ev1_phi, v1r, v1i, v2r, v2i,
     ev2_abs, ev2_phi, c1r, c1i, c2r, c2i) = np.loadtxt(evecs_file).T
    v1 = v1r + 1j*v1i
    v2 = v2r + 1j*v2i
    c1 = c1r + 1j*c1i
    c2 = c2r + 1j*c2i

    # configuration to delta
    # c = (-3.05, -1.65, -1.1, -0.55, 0.0, 0.55, 1.95)
    # v1 = np.concatenate([[1.0], v1, [0.0]])
    # v2 = np.concatenate([[0.0], v2, [1.0j]])
    # c1 = np.concatenate([[1j], c1, [0.0]])
    # c2 = np.concatenate([[1.0], c2, [1j]])

    # n, m = 0, len(c)

    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9.3, 3), dpi=220)

    ax1.set_title(r"$\arctan\vert c^i_1/c^i_2\vert$")
    ax1.plot(c, np.arctan(abs(v1/v2)), "o-", color=colors[0], clip_on=False)
    ax1.plot(c, np.arctan(abs(c1/c2)), "o-", color=colors[1], clip_on=False)

    eps_tm, delta_tm, v1_tm, v2_tm = get_toymodel_data()
    ax1.plot(delta_tm, np.arctan(abs(v1_tm[:, 0]/v1_tm[:, 1])), ls="--", color=colors[0])
    ax1.plot(delta_tm, np.arctan(abs(v2_tm[:, 0]/v2_tm[:, 1])), ls="--", color=colors[1])

    ax1.set_ylim(0, np.pi/2)
    ax1.set_xlabel(r"$\delta\cdot W$")

    ax2.set_title(r"$\operatorname{Arg} c^i_1/c^i_2$")
    ax2.plot(c, np.angle(v1/v2), "o-", color=colors[0], clip_on=False)
    ax2.plot(c, np.angle(c1/c2), "o-", color=colors[1], clip_on=False)

    exp_data = []
    for expn, (exp_c, exp_f) in enumerate(zip(exp_config, exp_file)):
        (_, _, v1r, v1i, v2r, v2i,
         _, _, c1r, c1i, c2r, c2i) = np.loadtxt(exp_f).T
        v1_exp = v1r + 1j*v1i
        v2_exp = v2r + 1j*v2i
        c1_exp = c1r + 1j*c1i
        c2_exp = c2r + 1j*c2i
        exp_data.append([c[exp_c], v1_exp, v2_exp, c1_exp, c2_exp])

    c_exp, v1_exp, v2_exp, c1_exp, c2_exp = np.asarray(exp_data).T

    ax1.plot(c_exp, np.arctan(abs(v1_exp/v2_exp)), "v:", color=colors[0], mfc="w", mec=colors[0], clip_on=False)
    ax1.plot(c_exp, np.arctan(abs(c1_exp/c2_exp)), "v:", color=colors[1], mfc="w", mec=colors[1], clip_on=False)
    ax2.plot(c_exp, np.angle(v1_exp/v2_exp), "v:", color=colors[0], mfc="w", mec=colors[0], clip_on=False)
    ax2.plot(c_exp, np.angle(c1_exp/c2_exp), "v:", color=colors[1], mfc="w", mec=colors[1], clip_on=False)

    # ax1.plot(c[4], 0.904979981018, "v-", color=colors[0], mfc="w", mec=colors[0], clip_on=False)
    # ax1.plot(c[4], 0.748095088658, "v-", color=colors[1], mfc="w", mec=colors[1], clip_on=False)
    # ax2.plot(c[4], -0.223734533323, "v-", color=colors[0], mfc="w", mec=colors[0], clip_on=False)
    # ax2.plot(c[4], 2.9830283703, "v-", color=colors[1], mfc="w", mec=colors[1], clip_on=False)

    # print np.arctan(abs(v1_exp/v2_exp))
    # print np.arctan(abs(c1_exp/c2_exp))
    # print np.angle(v1_exp/v2_exp)
    # print np.angle(c1_exp/c2_exp)

    ax2.plot(delta_tm, np.angle(v1_tm[:, 0]/v1_tm[:, 1]), ls="--", color=colors[0])
    ax2.plot(delta_tm, np.angle(v2_tm[:, 0]/v2_tm[:, 1]), ls="--", color=colors[1])
    ax2.set_xlabel(r"$\delta\cdot W$")

    # ax3.set_title(r"$\lambda$")
    # ax3.plot(c, (ev1_abs * np.exp(1j*ev1_phi)).real, marker="", ls="-", color=colors[0], clip_on=False)
    # ax3.plot(c, (ev1_abs * np.exp(1j*ev1_phi)).imag, marker="", ls="--", color=colors[0], clip_on=False)
    # ax3.plot(c, (ev2_abs * np.exp(1j*ev2_phi)).real, marker="", ls="-", color=colors[1], clip_on=False)
    # ax3.plot(c, (ev2_abs * np.exp(1j*ev2_phi)).imag, marker="", ls="--", color=colors[1], clip_on=False)
    # ax3.set_xlabel(r"$\delta\cdot W$")

    for a in (ax1, ax2): #, ax3):
        a.set_xlim(-3.1, 2.05)

    # plt.subplots_adjust(top=0.5)
    plt.tight_layout()

    if not save:
        plt.show()
    else:
        plt.savefig("coefficients.png")


if __name__ == '__main__':
    argh.dispatch_command(plot_coefficients)
