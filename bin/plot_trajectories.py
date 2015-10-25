#!/usr/bin/env python

from __future__ import division

from collections import namedtuple
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogFormatterMathtext
import numpy as np

from ep.helpers import map_trajectory
import ep.plot
from ep.waveguide import DirichletReduced, DirichletPositionDependentLossReduced


ep.plot.get_defaults()
colors, parula, _ = ep.plot.get_colors()

legend_kwargs = {'frameon': False,
        #'labelspacing': -0.25,
        #'columnspacing': 0.,
        #'handletextpad': -0.1,
        #'handlelength': 1.5,
        #'bbox_to_anchor': (0.0, 1.3, 1.0, 0.13),
        'bbox_to_anchor': (0.08, 0.87, 2., 0.087),
        'mode': 'expand',
        'ncol': 4}


def get_trajectories(ax1=None, ax2=None, wg_list=None, y_range_trajectory=None,
                     y_axis_step_length=5):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    s = WGam.adiabatic

    ax1.semilogy(x, abs(WGam.c0*s)**2, "-", color=colors[0], label=r"$|c_1|^2$")
    ax1.semilogy(x, abs(WGam.c1*s)**2, "-", color=colors[1], label=r"$|c_2|^2$")
    ax1.semilogy(x, abs(WGbm.c0*s)**2, "--", color=colors[0], label=r"$|c_1|^2$")
    ax1.semilogy(x, abs(WGbm.c1*s)**2, "--", color=colors[1], label=r"$|c_2|^2$")
    ax1.legend(loc="lower left", **legend_kwargs)

    ax2.semilogy(L - x, abs(WGbp.c1*s)**2, "-", color=colors[0], label=r"$|c_1|^2$")
    ax2.semilogy(L - x, abs(WGbp.c0*s)**2, "-", color=colors[1], label=r"$|c_2|^2$")
    ax2.semilogy(L - x, abs(WGap.c1*s)**2, "--", color=colors[0], label=r"$|c_1|^2$")
    ax2.semilogy(L - x, abs(WGap.c0*s)**2, "--", color=colors[1], label=r"$|c_2|^2$")

    for ax in (ax1, ax2):
        ax.set_yticks(10.**np.arange(0, -31, -y_axis_step_length))
        ax.tick_params(axis='y', which='minor', left='off', right='off')
        if y_range_trajectory:
            ax.set_ylim(*y_range_trajectory)

    # def custom_formatter_function(x, pos):
    #     if np.log(x) == 0:
    #         return "1  "
    #     else:
    #         return  r"$10^{}$".format(x)
    #
    # ax1.get_yaxis().set_major_formatter(FuncFormatter)

    # import ipdb; ipdb.set_trace()
    # labels = [label.get_text() for label in ax1.get_yticklabels()]
    # labels = ax1.get_yticks().tolist()
    # labels[0] = '1  '
    # ax1.set_yticklabels(labels)


def get_real_spectrum(ax1=None, ax2=None, wg_list=None, ms=5.0, mew=1.5,
                      fs='none', y_range_real_spectrum=None,
                      y_ticklabels_real_spectrum=None):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    s = WGam.adiabatic
    nstep = WGam.nstep

    ax1.plot(x, WGam.E0.real, "-", color=colors[0], label=r"Re $E_1$")
    ax1.plot(x, WGam.E1.real, "-", color=colors[1], label=r"Re $E_2$")
    ax1.plot(x[::nstep], map_trajectory(WGam.c0, WGam.c1,
             WGam.E0.real, WGam.E1.real)[::nstep], "k^",
             ms=ms)
    ax1.plot(x[nstep/2::nstep], map_trajectory(WGbm.c0, WGbm.c1,
             WGam.E0.real, WGam.E1.real)[nstep/2::nstep], "ks",
             ms=ms, mew=mew, fillstyle=fs)

    ax2.plot(L - x, WGap.E0.real, "-", color=colors[1], label=r"Re $E_1$")
    ax2.plot(L - x, WGap.E1.real, "-", color=colors[0], label=r"Re $E_2$")
    ax2.plot((L - x)[::nstep], map_trajectory(WGap.c0, WGap.c1,
             WGap.E0.real, WGap.E1.real)[::nstep], "ks",
             ms=ms, mew=mew, fillstyle=fs)
    ax2.plot((L - x)[nstep/2::nstep], map_trajectory(WGbp.c0, WGbp.c1,
             WGap.E0.real, WGap.E1.real)[nstep/2::nstep], "k^",
             ms=ms)
    energy_legend = copy.deepcopy(legend_kwargs)
    energy_legend.pop('mode')
    energy_legend.update({'ncol': 2,
                          'columnspacing': 0.75,
                          'bbox_to_anchor': (0.5, -0.075)})
    ax1.legend(loc="lower center", **energy_legend)
    ax2.legend(loc="lower center", **energy_legend)

    ax1.get_yaxis().set_tick_params(pad=2)

    for ax in (ax1, ax2):
        if y_range_real_spectrum:
            ax.set_ylim(*y_range_real_spectrum)
        if y_ticklabels_real_spectrum:
            ax.locator_params(axis='y', nbins=y_ticklabels_real_spectrum)


def get_imag_spectrum(ax1=None, ax2=None, wg_list=None,
                      y_range_imag_spectrum=None,
                      y_ticklabels_imag_spectrum=None):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    s = WGam.adiabatic

    ax1.plot(x, WGam.E0.imag, "-", color=colors[0], label=r"Im $E_1$")
    ax1.plot(x, WGam.E1.imag, "-", color=colors[1], label=r"Im $E_2$")

    ax2.plot(L - x, WGap.E0.imag, "-", color=colors[1], label=r"Im $E_1$")
    ax2.plot(L - x, WGap.E1.imag, "-", color=colors[0], label=r"Im $E_2$")

    energy_legend = copy.deepcopy(legend_kwargs)
    energy_legend.pop('mode')
    energy_legend.update({'ncol': 2,
                          'columnspacing': 0.75,
                          'bbox_to_anchor': (0.02, -0.075)})
    ax1.legend(loc="lower left", **energy_legend)
    ax2.legend(loc="lower left", **energy_legend)

    for ax in (ax1, ax2):
        if y_range_imag_spectrum:
            ax.set_ylim(*y_range_imag_spectrum)
        ax.locator_params(axis='y', nbins=4)
        if y_ticklabels_imag_spectrum:
            ax.locator_params(axis='y', nbins=y_ticklabels_imag_spectrum)


def plot_parameter_trajectory(figname=None, wg_kwargs=None, ep_coordinates=None,
                              pos_dep=False):
    f = plt.figure(figsize=(3.1, 3.1/3), dpi=220)
    ax = plt.gca()

    WGam = DirichletReduced(**wg_kwargs_am)
    eps, delta = WGam.get_cycle_parameters()
    ax.plot(delta, eps, color="k", lw=2.5, clip_on=False)
    if not ep_coordinates:
        x_EP, y_EP = WGam.x_EP, WGam.y_EP
    else:
        x_EP, y_EP = ep_coordinates

    if pos_dep:
        datay = [y_EP, -0.278, -0.234, -0.188, -0.140, -0.072, -0.048, -0.024, -0.007, 0.000, 0.0]
        datax = [x_EP,  0.033,  0.034,  0.034,  0.033,  0.028,  0.025,  0.021,  0.015, 0.008, -0.1]
        ax.plot(datay, datax, "k--", lw=0.75, dashes=[3, 3])
        datay = [y_EP, -0.352, -0.416, -0.506, -0.596, -0.650, -0.684, -0.736, -0.791, -0.827, -0.901, -0.972, -1.058, -1.145]
        datax = [x_EP,  0.030,  0.026,  0.018,  0.011,  0.007,  0.005,  0.002,  0.000,  0.000,  0.000,  0.003,  0.008,  0.015]
        ax.plot(datay, datax, "k--", lw=0.75, dashes=[1, 2])
        ax.annotate('EP', (-0.29, 0.041), textcoords='data',
                    weight='bold', size=14, color='black')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, eps.max())
    else:
        ax.plot([0, y_EP], [0.0, x_EP], "k--", lw=0.75, dashes=[3, 3])
        ax.plot([0, y_EP], [0.1, x_EP], "k--", lw=0.75, dashes=[1, 2])
        ax.plot(y_EP, x_EP, "o", color=colors[4], ms=7.5, mec='none', clip_on=False)
        ax.annotate('EP', (0.1, 0.04), textcoords='data',
                    weight='bold', size=14, color='black')

    ax.plot(delta[0], eps[0], "ko", ms=7.5, clip_on=False)
    ax.plot(delta[-1], eps[-1], "ko", ms=7.5, clip_on=False)
    ax.set_ylabel(r"Amplitude $\varepsilon$")
    ax.set_xlabel(r"Detuning $\delta$")
    ax.locator_params(axis='y', nbins=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')

    plt.savefig(figname, bbox_inches='tight')

def get_spectrum(D, wg_kwargs_am=None, fig_name_spectrum=None,
                 y_range_imag_spectrum=None, y_range_real_spectrum=None,
                 y_axis_step_length=5, y_ticklabels_real_spectrum=None,
                 y_ticklabels_imag_spectrum=None):

    wg_kwargs_bm = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_bm.update({'loop_direction': '-',
                         'init_state': 'b'})
    wg_kwargs_ap = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_ap.update({'loop_direction': '+',
                         'init_state': 'a'})
    wg_kwargs_bp = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_bp.update({'loop_direction': '+',
                         'init_state': 'b'})

    wg_kwarg_list = (wg_kwargs_am, wg_kwargs_bm, wg_kwargs_ap, wg_kwargs_bp)
    wg_list = [D(**wg_kwargs) for wg_kwargs in wg_kwarg_list]
    WGam, WGbm, WGap, WGbp = wg_list

    for w in wg_list:
        w.solve_ODE()
        print "...done."

    WG = namedtuple('WG', 'D x c0 c1 E0 E1 adiabatic nstep')
    adiabatic = WGam.Psi_adiabatic[:, 0]**(-1)
    if np.all(np.isnan(adiabatic)):
        adiabatic = 1.
    nstep = WGam.tN/10
    wg_list = [WG(wg, wg.t, wg.phi_a, wg.phi_b,
                  wg.eVals[:, 0], wg.eVals[:, 1],
                  adiabatic, nstep) for wg in wg_list]
    WGam, WGbm, WGap, WGbp = wg_list

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2,
                                               figsize=(6.2, 5.0/3.*2.), dpi=220,
                                               sharex=True, sharey=False)
    get_real_spectrum(ax1=ax1, ax2=ax2, wg_list=wg_list,
                      y_range_real_spectrum=y_range_real_spectrum,
                      y_ticklabels_real_spectrum=y_ticklabels_real_spectrum)
    get_imag_spectrum(ax1=ax3, ax2=ax4, wg_list=wg_list,
                      y_range_imag_spectrum=y_range_imag_spectrum,
                      y_ticklabels_imag_spectrum=y_ticklabels_imag_spectrum)

    for ax in (ax1, ax3):
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in (ax3, ax4):
        ax.set_xticks([0, WGam.D.L/2, WGam.D.L])
        ax.set_xticklabels([r"0", r"L/2", r"L"])

    for ax in (ax1, ax3):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    for ax in (ax2, ax4):
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')
        ax.set_yticklabels([])

    for n, ax in enumerate(f.get_axes()):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.tick_params(axis='both', which='minor', bottom='off',
                       left='off', right='off', top='off')

    f.text(0.5, -0., 'Spatial coordinate x', ha='center')
    f.text(-0.01, 0.94, 'a', weight='bold', size=14)
    f.text(-0.01, 0.49, 'b', weight='bold', size=14)

    plt.tight_layout(w_pad=0.8, h_pad=0.2)
    plt.subplots_adjust(hspace=0.2)

    plt.savefig(fig_name_spectrum, bbox_inches='tight')


def get_plot(D, wg_kwargs_am=None, fig_name_trajectories=None,
             y_range_trajectory=None, y_axis_step_length=5):

    wg_kwargs_bm = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_bm.update({'loop_direction': '-',
                         'init_state': 'b'})
    wg_kwargs_ap = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_ap.update({'loop_direction': '+',
                         'init_state': 'a'})
    wg_kwargs_bp = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_bp.update({'loop_direction': '+',
                         'init_state': 'b'})

    wg_kwarg_list = (wg_kwargs_am, wg_kwargs_bm, wg_kwargs_ap, wg_kwargs_bp)
    wg_list = [D(**wg_kwargs) for wg_kwargs in wg_kwarg_list]
    WGam, WGbm, WGap, WGbp = wg_list

    for w in wg_list:
        w.solve_ODE()
        print "...done."

    WG = namedtuple('WG', 'D x c0 c1 E0 E1 adiabatic nstep')
    adiabatic = WGam.Psi_adiabatic[:, 0]**(-1)
    if np.all(np.isnan(adiabatic)):
        adiabatic = 1.
    nstep = WGam.tN/10
    wg_list = [WG(wg, wg.t, wg.phi_a, wg.phi_b,
                  wg.eVals[:, 0], wg.eVals[:, 1],
                  adiabatic, nstep) for wg in wg_list]
    WGam, WGbm, WGap, WGbp = wg_list

    print
    print abs(WGam.c0[-1]/WGam.c1[-1])**-2
    print abs(WGap.c0[-1]/WGap.c1[-1])**2
    print abs(WGbm.c0[-1]/WGbm.c1[-1])**-2
    print abs(WGbp.c0[-1]/WGbp.c1[-1])**2
    print
    print abs(WGam.c0[-1]/WGam.c1[-1])**2
    print abs(WGap.c0[-1]/WGap.c1[-1])**-2
    print abs(WGbm.c0[-1]/WGbm.c1[-1])**2
    print abs(WGbp.c0[-1]/WGbp.c1[-1])**-2

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2,
                                               figsize=(6.2, 5.0/3.*2.), dpi=220,
                                               sharex=True, sharey=False)
    get_trajectories(ax1=ax1, ax2=ax2, wg_list=wg_list,
                     y_range_trajectory=y_range_trajectory,
                     y_axis_step_length=y_axis_step_length)

    for ax in (ax3, ax4):
        ax.set_xticks([0, WGam.D.L/2, WGam.D.L])
        ax.set_xticklabels([r"0", r"L/2", r"L"])

    for ax in (ax1, ax3):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    for ax in (ax2, ax4):
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')
        ax.set_yticklabels([])

    for n, ax in enumerate(f.get_axes()):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.tick_params(axis='both', which='minor', bottom='off',
                       left='off', right='off', top='off')

    f.text(0.5, -0., 'Spatial coordinate x', ha='center')
    f.text(-0.01, 0.94, 'a', weight='bold', size=14)
    f.text(-0.01, 0.49, 'b', weight='bold', size=14)

    plt.tight_layout(w_pad=0.8, h_pad=0.2)
    plt.subplots_adjust(hspace=0.2)

    plt.savefig(fig_name_trajectories, bbox_inches='tight')


if __name__ == '__main__':
    wg_kwargs_am = {
            # pos. dep. loss configuration/ works for (approx.) uniform loss (scaled with eps^2)
            # use N=2.05 to fit simulation setup
            'N': 2.05,
            'loop_type': 'Bell',
            'loop_direction': '-',
            'init_state': 'a',
            'init_state_method': 'energy',
            'W': 1,
            'L': 1,
            'eta':  0.6,
            'eta0': 0.0,
            'x_R0': 0.1,
            'y_R0': 0.85,
            'init_phase': 0.3,
            'switch_losses_on_off': True
            }
    get_plot(DirichletReduced, wg_kwargs_am,
             "constant_cn_vs_x_reduced_trajectory.pdf",
             y_axis_step_length=10,
             y_range_trajectory=[5e-34, 1e2])

    get_spectrum(DirichletReduced, wg_kwargs_am,
                 "constant_cn_vs_x_reduced_spectrum.pdf",
                 y_range_real_spectrum=[-1.2, 1.2],
                 y_ticklabels_real_spectrum=3,
                 y_ticklabels_imag_spectrum=5,
                 y_range_imag_spectrum=[-1.6, 0.1])

    plot_parameter_trajectory(wg_kwargs=wg_kwargs_am,
                              figname="constant_parameter_space_reduced_test.pdf")

