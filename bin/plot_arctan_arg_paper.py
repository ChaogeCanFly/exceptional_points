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
def plot_coefficients(save=False, evecs_file="evecs_t.dat", exp_file=None, exp_config=None, lw=1.0):
    """docstring for plot_coefficients"""

    c = (-1.65, -1.1, -0.55, 0.0, 0.55)

    (ev1_abs, ev1_phi, v1r, v1i, v2r, v2i,
     ev2_abs, ev2_phi, c1r, c1i, c2r, c2i) = np.loadtxt(evecs_file).T
    v1 = v1r + 1j*v1i
    v2 = v2r + 1j*v2i
    c1 = c1r + 1j*c1i
    c2 = c2r + 1j*c2i

    # plot
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6.3, 2), dpi=220)

    f.text(0.0, 0.9, 'a', weight='bold', size=12)
    f.text(0.5, 0.9, 'b', weight='bold', size=12)

    exp_data = []
    exp_data_evals = []
    for expn, (exp_c, exp_f) in enumerate(zip(exp_config, exp_file)):
        (ev1_abs_exp, _, v1r, v1i, v2r, v2i,
         ev2_abs_exp, _, c1r, c1i, c2r, c2i) = np.loadtxt(exp_f).T
        v1_exp = v1r + 1j*v1i
        v2_exp = v2r + 1j*v2i
        c1_exp = c1r + 1j*c1i
        c2_exp = c2r + 1j*c2i
        exp_data.append([c[exp_c], v1_exp, v2_exp, c1_exp, c2_exp])
        exp_data_evals.append([ev1_abs_exp, ev2_abs_exp])

    c_exp, v1_exp, v2_exp, c1_exp, c2_exp = np.asarray(exp_data).T
    ev1_abs_exp, ev2_abs_exp = np.asarray(exp_data_evals).T

    ax1.set_ylabel(r"$\arctan\vert c^n_1/c^n_2\vert$")
    ax1.plot(c, np.arctan(abs(v1/v2)), "o-", color=colors[0], clip_on=False, mec='none', lw=lw)
    ax1.plot(c, np.arctan(abs(c1/c2)), "o-", color=colors[1], clip_on=False, mec='none', lw=lw)

    eps_tm, delta_tm, v1_tm, v2_tm = get_toymodel_data()
    ax1.plot(delta_tm, np.arctan(abs(v1_tm[:, 0]/v1_tm[:, 1])), ls="--", color=colors[0], lw=lw)
    ax1.plot(delta_tm, np.arctan(abs(v2_tm[:, 0]/v2_tm[:, 1])), ls="--", color=colors[1], lw=lw)

    ax1.set_xlabel(r"$\delta\cdot W$")
    ax1.plot(c_exp, np.arctan(abs(v1_exp/v2_exp)), "v:", color=colors[0], mfc="w", mec=colors[0], clip_on=False, lw=lw)
    ax1.plot(c_exp, np.arctan(abs(c1_exp/c2_exp)), "v:", color=colors[1], mfc="w", mec=colors[1], clip_on=False, lw=lw)

    ax1.set_ylim(0, np.pi/2)
    ax1.set_yticks([0, np.pi/4, np.pi/2])
    ax1.set_yticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])

    ax2.set_ylabel(r"$\vert \tau_1/\tau_2 \vert$")
    ax2.semilogy(c, ev1_abs/ev2_abs, marker="o", ls="-", color="k", clip_on=False, mec='none', lw=lw)
    ax2.semilogy(c_exp, ev1_abs_exp/ev2_abs_exp, marker="v", ls=":", color="k", mfc="w", mec="k", clip_on=False, lw=lw)
    ax2.set_ylim(1e0, 1e3)
    ax2.set_xlabel(r"$\delta\cdot W$")

    for ax in (ax1, ax2): #, ax3):
        ax.set_xlim(-3.1, 2.05)
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
