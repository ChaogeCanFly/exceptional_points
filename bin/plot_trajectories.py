#!/usr/bin/env python

from __future__ import division

from collections import import namedtuple
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import import LogNorm
from matplotlib.ticker import import MultipleLocator, FormatStrFormatter
import multiprocessing
import numpy as np

from ep.helpers import import map_trajectory
import ep.plot
from ep.waveguide import import DirichletPositionDependentLoss, Dirichlet


ep.plot.get_defaults()
colors, parula, _ = ep.plot.get_colors()


def plot_trajectories_energy(D, wg_kwargs_am=None, fig_name_trajectories=None, fig_name_parameter_space=None,
                             y_range_trajectory=None, y_range_imag_spectrum=None, y_range_real_spectrum=None,
                             y_axis_step_length=5, y_ticklabels_real_spectrum=None, y_ticklabels_imag_spectrum=None,
                             ms=7.5, mew=1.5, fs='none', ep_coordinates=None, pos_dep=False):

    legend_kwargs = {'frameon': False,
                     #'labelspacing': -0.25,
                     #'columnspacing': 0.,
                     #'handletextpad': -0.1,
                     #'handlelength': 1.5,
                     #'bbox_to_anchor': (0.0, 1.3, 1.0, 0.13),
                     'bbox_to_anchor': (0.08, 0.87, 2., 0.087),
                     'mode': 'expand',
                     'ncol': 4}

    wg_kwargs_bm = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_bm.update({'loop_direction': '-',
                         'init_state': 'b'})
    wg_kwargs_ap = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_ap.update({'loop_direction': '+',
                         'init_state': 'a'})
    wg_kwargs_bp = copy.deepcopy(wg_kwargs_am)
    wg_kwargs_bp.update({'loop_direction': '+',
                         'init_state': 'b'})

    WGam = D(**wg_kwargs_am)
    WGbm = D(**wg_kwargs_bm)
    WGap = D(**wg_kwargs_ap)
    WGbp = D(**wg_kwargs_bp)

    wg_list = (WGam, WGbm, WGap, WGbp)
    kwd_list = (wg_kwargs_am, wg_kwargs_bm, wg_kwargs_ap, wg_kwargs_bp)
    for w in wg_list:
        w.solve_ODE()
        print "...done."

    WG = namedtuple('WG', 'D x c0 c1 E0 E1')
    WGam, WGbm, WGap, WGbp = [WG(wg, wg.t, wg.phi_a, wg.phi_b,
                                 wg.eVals[:, 0], wg.eVals[:, 1]) for wg in wg_list]

    # settings
    x = WGam.x
    L = WGam.D.L
    nstep = WGam.D.tN/10
    s = WGam.D.Psi_adiabatic[:, 0]**(-1)
    if np.all(np.isnan(s)):
        s = 1.

    # plots
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = subplots(ncols=2, nrows=3, 
                                                       figsize=(6.2, 5.0), dpi=220,
                                                       sharex=True, sharey=False)

    ax1.semilogy(x, abs(WGam.c0*s)**2, "-", color=colors[0], label=r"$|c_1|^2$")
    ax1.semilogy(x, abs(WGam.c1*s)**2, "-", color=colors[1], label=r"$|c_2|^2$")
    ax1.semilogy(x, abs(WGbm.c0*s)**2, "--", color=colors[0], label=r"$|c_1|^2$")
    ax1.semilogy(x, abs(WGbm.c1*s)**2, "--", color=colors[1], label=r"$|c_2|^2$")
    #ax1.semilogy(x[::nstep/5], abs(WGbm.c0*s)[::nstep/5]**2, "o", mec=colors[0], ms=2., color=colors[0], label=r"$|c_1|^2$")
    #ax1.semilogy(x[::nstep/5], abs(WGbm.c1*s)[::nstep/5]**2, "o", mec=colors[1], ms=2., color=colors[1], label=r"$|c_2|^2$")
    ax1.legend(loc="lower left", **legend_kwargs)

    ax2.semilogy(L - x, abs(WGbp.c1*s)**2, "-", color=colors[0], label=r"$|c_1|^2$")
    ax2.semilogy(L - x, abs(WGbp.c0*s)**2, "-", color=colors[1], label=r"$|c_2|^2$")
    ax2.semilogy(L - x, abs(WGap.c1*s)**2, "--", color=colors[0], label=r"$|c_1|^2$")
    ax2.semilogy(L - x, abs(WGap.c0*s)**2, "--", color=colors[1], label=r"$|c_2|^2$")

    ax3.plot(x, WGam.E0.real, "-", color=colors[0], label=r"Re $E_1$")
    ax3.plot(x, WGam.E1.real, "-", color=colors[1], label=r"Re $E_2$")
    ax3.plot(x[::nstep], map_trajectory(WGam.c0, WGam.c1,
                                        WGam.E0.real, WGam.E1.real)[::nstep], "k^",
                                        ms=ms)
    ax3.plot(x[nstep/2::nstep], map_trajectory(WGbm.c0, WGbm.c1,
                                               WGam.E0.real, WGam.E1.real)[nstep/2::nstep], "ks",
                                               ms=ms, mew=mew, fillstyle=fs) 

    ax4.plot(L - x, WGap.E0.real, "-", color=colors[1], label=r"Re $E_1$")
    ax4.plot(L - x, WGap.E1.real, "-", color=colors[0], label=r"Re $E_2$")
    ax4.plot((L - x)[::nstep], map_trajectory(WGap.c0, WGap.c1,
                                              WGap.E0.real, WGap.E1.real)[::nstep], "ks",
                                              ms=ms, mew=mew, fillstyle=fs)
    ax4.plot((L - x)[nstep/2::nstep], map_trajectory(WGbp.c0, WGbp.c1,
                                                     WGap.E0.real, WGap.E1.real)[nstep/2::nstep], "k^",
                                                     ms=ms)

    ax5.plot(x, WGam.E0.imag, "-", color=colors[0], label=r"Im $E_1$")
    ax5.plot(x, WGam.E1.imag, "-", color=colors[1], label=r"Im $E_2$")

    ax6.plot(L - x, WGap.E0.imag, "-", color=colors[1], label=r"Im $E_1$")
    ax6.plot(L - x, WGap.E1.imag, "-", color=colors[0], label=r"Im $E_2$")

    energy_legend = copy.deepcopy(legend_kwargs)
    energy_legend.pop('mode')
    energy_legend.update({'ncol': 2,
                         #'labelspacing': -0.25,
                         'columnspacing': 0.75,
                         #'handletextpad': -0.1,
                         #'handlelength': 1.5,
                         #'bbox_to_anchor': (0.0, 1.3, 1.0, 0.13),
                          #'handlelength': 1.5,
                          'bbox_to_anchor': (0.5, -0.075)
                         })
    ax3.legend(loc="lower center", **energy_legend)
    ax4.legend(loc="lower center", **energy_legend)
    energy_legend.update({'ncol': 2,
                          'bbox_to_anchor': (0.02, -0.075)
                         })
    ax5.legend(loc="lower left", **energy_legend)
    ax6.legend(loc="lower left", **energy_legend)

    for ax in (ax1, ax2):
        ax.set_yticks(10.**np.arange(0, -31, -y_axis_step_length))

    for ax in (ax3, ax4, ax5, ax6):
        ax.locator_params(axis='y', nbins=3)

    #for ax in (ax1, ax2, ax3, ax4):
    #    ax.set_xticklabels([])

    for ax in (ax5, ax6):
        ax.set_xlim(0, L)
        ax.locator_params(axis='x', nbins=5)
        #ax.set_xlabel(r"$x$", labelpad=-2.)
    for ax in f.get_axes():
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.tick_params(axis='both', which='minor', bottom='off', left='off', right='off', top='off')
        #ax.tick_params(which='major', length=2.5)

    for ax in (ax1, ax3, ax5):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    for ax in (ax2, ax4, ax6):
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')

    for ax in (ax2, ax4, ax6):
        #ax.yaxis.tick_right()
        ax.set_yticklabels([])

    tight_layout(w_pad=0.8, h_pad=0.2)
    #gcf().subplots_adjust(bottom=0.5)

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    labels[0] = '1      '
    labels[0] = '1  '
    ax1.set_yticklabels(labels)
    for ax in (ax1, ax2):
        ax.tick_params(axis='y', which='minor', left='off', right='off')

    if y_range_trajectory:
        for ax in (ax1, ax2):
            ax.set_ylim(*y_range_trajectory)

    if y_range_real_spectrum:
        for ax in (ax3, ax4):
            ax.set_ylim(*y_range_real_spectrum)

    if y_range_imag_spectrum:
        for ax in (ax5, ax6):
            ax.set_ylim(*y_range_imag_spectrum)

    for ax in (ax3, ax5):
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in (ax5, ax6):
        ax.set_xticks([0, L/2, L])
        ax.set_xticklabels([r"0", r"L/2", r"L"])


    if y_ticklabels_real_spectrum:
        ax3.locator_params(axis='y', nbins=y_ticklabels_real_spectrum)

    if y_ticklabels_imag_spectrum:
        ax5.locator_params(axis='y', nbins=y_ticklabels_imag_spectrum)

    f.text(0.5, -0., 'Spatial coordinate x', ha='center')
    f.text(-0.01, 0.94, 'a', weight='bold', size=14)
    f.text(-0.01, 0.64, 'b', weight='bold', size=14)
    f.text(-0.01, 0.34, 'c', weight='bold', size=14)

    ax1.get_yaxis().set_tick_params(pad=2)

    savefig(fig_name_trajectories, bbox_inches='tight')

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

    f = figure(figsize=(3.1, 3.1/3), dpi=220)
    ax = gca()
    eps, delta = WGam.D.get_cycle_parameters(WGam.x)
    #xticks(rotation=0) #45)
    ax.plot(delta, eps, color="k", lw=2.5, clip_on=False)
    if not ep_coordinates:
        x_EP, y_EP = WGam.D.x_EP, WGam.D.y_EP
    else:
        x_EP, y_EP = ep_coordinates

    if pos_dep:
        datay = [y_EP, -0.278, -0.234, -0.188, -0.140, -0.072, -0.048, -0.024, -0.007, 0.000, 0.0]
        datax = [x_EP,  0.033,  0.034,  0.034,  0.033,  0.028,  0.025,  0.021,  0.015, 0.008, -0.1]
        ax.plot(datay, datax, "k--", lw=0.75, dashes=[3, 3])
        datay = [y_EP, -0.352, -0.416, -0.506, -0.596, -0.650, -0.684, -0.736, -0.791, -0.827, -0.901, -0.972, -1.058, -1.145]
        datax = [x_EP,  0.030,  0.026,  0.018,  0.011,  0.007,  0.005,  0.002,  0.000,  0.000,  0.000,  0.003,  0.008,  0.015]
        ax.plot(datay, datax, "k--", lw=0.75, dashes=[1, 2])
        #ax.set_xlim(-1.1*delta.min(), 1.1*delta.max())
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, eps.max())
        ax.annotate('EP', (-0.29, 0.041), textcoords='data',
                    weight='bold', size=14, color='black')
    else:
        ax.plot([0, y_EP], [0.0, x_EP], "k--", lw=0.75, dashes=[3, 3])
        ax.plot([0, y_EP], [0.1, x_EP], "k--", lw=0.75, dashes=[1, 2])
        ax.annotate('EP', (0.1, 0.04), textcoords='data',
                    weight='bold', size=14, color='black') 

    ax.plot(y_EP, x_EP, "o", color=colors[4], ms=7.5, mec='none', clip_on=False)
    ax.plot(delta[0], eps[0], "ko", ms=7.5, clip_on=False)
    ax.plot(delta[-1], eps[-1], "ko", ms=7.5, clip_on=False)
    ax.set_ylabel(r"Amplitude $\varepsilon$") #/W$")
    ax.set_xlabel(r"Detuning $\delta$") #, labelpad=-3) # \cdot W$")
    ax.locator_params(axis='y', nbins=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_tick_params(pad=7.5)
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')
    savefig(fig_name_parameter_space, bbox_inches='tight')

    return WGam, WGbm, WGap, WGbp


if __name__ == '__main__':
     plot_trajectories_energy()
