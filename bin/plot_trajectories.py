#!/usr/bin/env python

from __future__ import division

from collections import namedtuple
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogFormatterMathtext
import numpy as np

import argh
from PIL import Image

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
        # 'bbox_to_anchor': (0.08, 0.87, 2., 0.087),
        'bbox_to_anchor': (0.08, 0.87, 2.0, 0.087),
        'mode': 'expand',
        'ncol': 4}


def plot_png(fig=None):
    img_1, img_2 = [Image.open(png) for png in ('waveguide_arrow.001_resize.png', 'waveguide_arrow.002_resize.png')]
    img_1, img_2 = [np.array(img).astype(np.float)/255. for img in (img_1, img_2)]
    fig
    fig.figimage(img_1, 130, 375, zorder=20, cmap='Greys_r')
    fig.figimage(img_2, 425, 375, zorder=20, cmap='Greys_r')
    plt.subplots_adjust(top=0.9)


def get_trajectories(ax1=None, ax2=None, ax3=None, ax4=None,
                     wg_list=None, y_range_trajectory=None,
                     y_axis_step_length=5):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    s = WGam.adiabatic

    ax1.semilogy(x, abs(WGam.c0*s)**2, "-", color=colors[0])
    ax1.semilogy(x, abs(WGam.c1*s)**2, "-", color=colors[1])
    ax3.semilogy(x, abs(WGbm.c0*s)**2, "-", color=colors[0])
    ax3.semilogy(x, abs(WGbm.c1*s)**2, "-", color=colors[1])

    ax2.semilogy(L - x, abs(WGbp.c1*s)**2, "-", color=colors[0])
    ax2.semilogy(L - x, abs(WGbp.c0*s)**2, "-", color=colors[1])
    ax4.semilogy(L - x, abs(WGap.c1*s)**2, "-", color=colors[0])
    ax4.semilogy(L - x, abs(WGap.c0*s)**2, "-", color=colors[1])

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_yticks(10.**np.arange(0, -31, -y_axis_step_length))
        ax.tick_params(axis='y', which='minor', left='off', right='off')
        if y_range_trajectory:
            ax.set_ylim(*y_range_trajectory)

    ax1.set_ylabel(r"Populations $|c_i|^2$")
    ax3.set_ylabel(r"Populations $|c_i|^2$")

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


def get_trajectories_new(ax1=None, ax2=None,
                         wg_list=None, y_range_trajectory=None,
                         y_axis_step_length=5):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    s = WGam.adiabatic

    norm_1 = abs(WGam.c0*s)**2 + abs(WGam.c1*s)**2
    norm_2 = abs(WGbm.c0*s)**2 + abs(WGbm.c1*s)**2
    norm_3 = abs(WGbp.c0*s)**2 + abs(WGbp.c1*s)**2
    norm_4 = abs(WGap.c0*s)**2 + abs(WGap.c1*s)**2

    ax1.semilogy(x, norm_1, "--", color=colors[0])
    ax1.semilogy(x, norm_2, "--", color=colors[1])
    ax2.semilogy(L - x, norm_3, "--", color=colors[0])
    ax2.semilogy(L - x, norm_4, "--", color=colors[1])

    for ax in (ax1, ax2):
        ax.set_yticks(10.**np.arange(0, -31, -y_axis_step_length))
        ax.tick_params(axis='y', which='minor', left='off', right='off')
        if y_range_trajectory:
            ax.set_ylim(*y_range_trajectory)

    ax1.set_ylabel(r"Norm") # $|c_1|^2 + |c_2|^2$")


def get_real_spectrum(ax1=None, ax2=None, wg_list=None, ms=5.0, mew=1.5,
                      fs='none', y_range_real_spectrum=None,
                      y_ticklabels_real_spectrum=None, projection=False):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    nstep = WGam.nstep

    ax1.plot(x, WGam.E0.real, "-", color=colors[0]) #, label=r"Re $E_1$")
    ax1.plot(x, WGam.E1.real, "-", color=colors[1]) #, label=r"Re $E_2$")
    if projection:
        ax1.plot(x[::nstep], map_trajectory(WGam.c0, WGam.c1,
                WGam.E0.real, WGam.E1.real)[::nstep], "k^",
                ms=ms)
        ax1.plot(x[nstep/2::nstep], map_trajectory(WGbm.c0, WGbm.c1,
                WGam.E0.real, WGam.E1.real)[nstep/2::nstep], "ks",
                ms=ms, mew=mew, fillstyle=fs)
    ax1.set_ylabel(r"Real spectrum", labelpad=0)

    if ax2:
        ax2.plot(L - x, WGap.E0.real, "-", color=colors[1])
        ax2.plot(L - x, WGap.E1.real, "-", color=colors[0])
        if projection:
            ax2.plot((L - x)[::nstep], map_trajectory(WGap.c0, WGap.c1,
                    WGap.E0.real, WGap.E1.real)[::nstep], "ks",
                    ms=ms, mew=mew, fillstyle=fs)
            ax2.plot((L - x)[nstep/2::nstep], map_trajectory(WGbp.c0, WGbp.c1,
                    WGap.E0.real, WGap.E1.real)[nstep/2::nstep], "k^",
                    ms=ms)

    _, delta = WGam.D.get_cycle_parameters()
    k1 = WGam.D.k0
    kb = WGam.D.kr + delta
    z1 = WGap.E0.real[-1] + 0*x + (kb[-1] - kb[0])
    # z1 = WGap.E0.real[-1] + 0*x + (delta[-1] - delta[0])
    z2 = WGap.E0.real[-1] + 0*x
    ax1.plot(x, z1, "k--", lw=0.75)
    ax1.plot(x, z2, "k--", lw=0.75)
    ax1.plot(x, 0*z2, "k--", lw=0.75)
    print z1[0]
    print z2[0]
    # ax1.annotate('', xy=(L*0.95, z2[0]*1.10), xycoords='data',
    #              xytext=(L*0.95, z1[0]*1.05), textcoords='data',
    #              arrowprops={'arrowstyle': '<->'})
    # ax1.annotate(r'$k_b(L) - k_b(0)$', (0.575*L, 0.25*abs(kb[0] - kb[-1])/2.),
    #              textcoords='data')
    ax1.annotate('', xy=(L*0.99, z2[0]*1.10), xycoords='data',
                 xytext=(L*0.99, z1[0]*1.05), textcoords='data',
                 arrowprops={'arrowstyle': '<->'}, annotation_clip=False)
    ax1.annotate(r'$k_b(L) - k_b(0)$', (1.01, 0.675),
                 textcoords='axes fraction', rotation=90, annotation_clip=False)

    # ax1.get_yaxis().set_tick_params(pad=2)

    for ax in (ax1, ax2):
        if ax:
            if y_range_real_spectrum:
                ax.set_ylim(*y_range_real_spectrum)
            if y_ticklabels_real_spectrum:
                ax.locator_params(axis='y', nbins=y_ticklabels_real_spectrum)


def get_real_K(ax1=None, ax2=None, wg_list=None, ms=5.0, mew=1.5,
               fs='none', y_range_real_spectrum=None,
               y_ticklabels_real_spectrum=None, projection=False):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    nstep = WGam.nstep
    k1 = WGam.D.k0
    k2 = WGam.D.k1

    eps, delta = WGam.D.get_cycle_parameters()
    kb = WGam.D.kr + delta

    ax1.plot(x, (k1 + delta - WGam.E0.real) % kb, "-", color=colors[0])
    ax1.plot(x, (k1 + delta - WGam.E1.real) % kb, "-", color=colors[1])
    ax1.set_ylabel(r"Real Bloch wavenumber")

    ax1.plot(x, k1 - kb[-1] + 0*x, "k--", lw=0.75)
    ax1.plot(x, k1 - kb[0] + 0*x, "k--", lw=0.75)
    ax1.annotate('', xy=(L*0.95, (k1 - kb[-1])*0.825), xycoords='data',
                 xytext=(L*0.95, (k1 - kb[0])*1.025), textcoords='data',
                 arrowprops={'arrowstyle': '<->'})
    ax1.annotate(r'$k_b(L) - k_b(0)$', xy=(0.28*L, 1.25*abs(kb[0] - kb[-1])/2.),
                 xycoords='data', xytext=(5, 0), textcoords='data')

    if ax2:
        ax2.plot(L - x, (k1 + delta - WGap.E0.real) % kb, "-", color=colors[1])
        ax2.plot(L - x, (k1 + delta - WGap.E1.real) % kb, "-", color=colors[0])

    for ax in (ax1, ax2):
        if ax:
            if y_range_real_spectrum:
                ax.set_ylim(*y_range_real_spectrum)
            if y_ticklabels_real_spectrum:
                ax.locator_params(axis='y', nbins=y_ticklabels_real_spectrum)


def get_real_spectrum_new(ax1=None, ax2=None, wg_list=None, ms=5.0, mew=1.5,
                          fs='none', y_range_real_spectrum=None,
                          y_ticklabels_real_spectrum=None, projection=False):
    WGam, WGbm, WGap, WGbp = wg_list
    x = WGam.x
    L = WGam.D.L
    nstep = WGam.nstep

    lw = 0.75

    ax1.plot(x, WGam.E0.real, "k-", lw=lw)
    ax1.plot(x, WGam.E1.real, "k-", lw=lw)
    if projection:
        ax1.plot(x, map_trajectory(WGam.c0, WGam.c1,
                WGam.E0.real, WGam.E1.real), "--", color=colors[0],
                ms=ms)
        ax1.plot(x, map_trajectory(WGbm.c0, WGbm.c1,
                WGam.E0.real, WGam.E1.real), "--", color=colors[1],
                ms=ms, mew=mew, fillstyle=fs)
    ax1.set_ylabel(r"Real spectrum")
    # ax1.get_yaxis().set_tick_params(pad=5)

    if ax2:
        ax2.plot(L - x, WGap.E0.real, "k-", lw=lw)
        ax2.plot(L - x, WGap.E1.real, "k-", lw=lw)
        if projection:
            ax2.plot((L - x), map_trajectory(WGap.c0, WGap.c1,
                    WGap.E0.real, WGap.E1.real), "--", color=colors[1],
                    ms=ms, mew=mew, fillstyle=fs)
            ax2.plot((L - x), map_trajectory(WGbp.c0, WGbp.c1,
                    WGap.E0.real, WGap.E1.real), "--", color=colors[0],
                    ms=ms)

    for ax in (ax1, ax2):
        if ax:
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

    ax1.plot(x, WGam.E0.imag, "-", color=colors[0])
    ax1.plot(x, WGam.E1.imag, "-", color=colors[1])
    # ax1.set_ylabel(r"Imaginary spectrum $\mathrm{Im} E_n$")
    ax1.set_ylabel(r"Imaginary spectrum")
    ax1.plot(x, 0*x, "k--", lw=0.75)

    if ax2:
        ax2.plot(L - x, WGap.E0.imag, "-", color=colors[1])
        ax2.plot(L - x, WGap.E1.imag, "-", color=colors[0])

    for ax in (ax1, ax2):
        if ax:
            if y_range_imag_spectrum:
                ax.set_ylim(*y_range_imag_spectrum)
            ax.locator_params(axis='y', nbins=4)
            if y_ticklabels_imag_spectrum:
                ax.locator_params(axis='y', nbins=y_ticklabels_imag_spectrum)


def plot_parameter_trajectory(figname=None, wg=None, ep_coordinates=None,
                              pos_dep=False):
    WGam = wg
    eps, delta = WGam.get_cycle_parameters()

    f = plt.figure(figsize=(3.1, 3.1/2.5), dpi=220)
    ax = plt.gca()
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
                    weight='bold', size=12, color='black')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, eps.max())
    else:
        xline = np.linspace(x_EP, 0.099, 16)
        yline = 0.*xline
        ax.plot([0, y_EP], [0.0, x_EP], "k--", lw=0.75, dashes=[3, 3])
        # ax.plot([0, y_EP], [0.1, x_EP], "k--", lw=0.75, dashes=[1, 1])
        ax.plot(yline, xline, "ko", ms=0.5, lw=0.75)
        ax.plot(y_EP, x_EP, "o", color=colors[4], ms=7.5, mec='none', clip_on=False)
        ax.annotate('EP', (0.1, 0.04), textcoords='data',
                    weight='bold', size=12, color='black')

    ax.plot(delta[0], eps[0], "ko", ms=7.5, clip_on=False)
    ax.plot(delta[-1], eps[-1], "ko", ms=7.5, clip_on=False)
    ax.set_ylabel(r"Amplitude $\sigma$")
    ax.set_xlabel(r"Detuning $\delta$")
    ax.locator_params(axis='y', nbins=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')

    plt.savefig(figname, bbox_inches='tight')


def plot_parameter_trajectory_p1_p2(figname=None, wg=None, ep_coordinates=None,
                                    pos_dep=False):
    WGam = wg
    eps, delta = WGam.get_cycle_parameters()

    f = plt.figure(figsize=(3.1, 3.1/2.5), dpi=220)
    ax = plt.gca()

    D = lambda d: 0.25*(1. + np.cos(np.pi*(d - WGam.init_phase)/WGam.y_R0))**2
    phi = lambda e, d: np.arctan(e/d) - np.pi*0.5*(np.sign(np.arctan(e/d)) - 1.)
    get_p1 = lambda e, d: ((e/WGam.x_R0)**1 - 0.01*D(d))*np.cos(2.*phi(e, d))
    get_p2 = lambda e, d: ((e/WGam.x_R0)**1 - 0.01*D(d))*np.sin(2.*phi(e, d))
    p1 = get_p1(eps, delta)
    p2 = get_p2(eps, delta)

    ax.plot(p1, p2, color="k", lw=0.5, clip_on=False)
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
                    weight='bold', size=12, color='black')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, eps.max())
    else:
        # xline = np.linspace(x_EP, 0.099, 16)
        # epsline = np.linspace(0.0, x_EP, 100)
        # epsline = np.linspace(0.0, WGam.x_EP, 100)
        epsline = np.linspace(WGam.x_EP, WGam.x_R0, 100)
        print epsline.max()
        print eps.max()
        print get_p1(0.1, 1e-20)
        print get_p2(0.1, 1e-20)
        deltaline = 0.*epsline
        p1_EP = get_p1(x_EP, y_EP)
        p2_EP = get_p2(x_EP, y_EP)
        print "x_EP, y_EP", x_EP, y_EP
        # ax.plot([0, y_EP], [0.0, x_EP], "k--", lw=0.75, dashes=[3, 3])
        # ax.plot([0, y_EP], [0.1, x_EP], "k--", lw=0.75, dashes=[1, 1])
        p1line = get_p1(epsline, deltaline)
        p2line = get_p2(epsline, deltaline)
        ax.plot(p1line, p2line, "r-", ms=0.5, lw=0.75)
        ax.plot(p1_EP, p2_EP, "o", color=colors[4], ms=2.5, mec='none', clip_on=False)
        print "x_EP, y_EP", p1line[-1], p2line[-1]
        # ax.plot(p1line[-1], p2line[-1], "o", color=colors[4], ms=2.5, mec='none', clip_on=False)
        # ax.annotate('EP', (0.1, 0.04), textcoords='data',
        #             weight='bold', size=12, color='black')
        plt.show()

    # ax.plot(delta[0], eps[0], "ko", ms=7.5, clip_on=False)
    # ax.plot(delta[-1], eps[-1], "ko", ms=7.5, clip_on=False)
    ax.set_ylabel(r"Amplitude $\sigma$")
    ax.set_xlabel(r"Detuning $\delta$")
    ax.locator_params(axis='y', nbins=4)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')

    plt.savefig(figname, bbox_inches='tight')


def plot_spectrum(wg_list=None, figname=None,
                  y_range_imag_spectrum=None, y_range_real_spectrum=None,
                  y_axis_step_length=5, y_ticklabels_real_spectrum=None,
                  y_ticklabels_imag_spectrum=None, projection=False):

    WGam, WGbm, WGap, WGbp = wg_list

    f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1,
                                 figsize=(6.2, 5.0/2.5*1.), dpi=220,
                                 sharex=True, sharey=False)
    get_real_spectrum(ax1=ax1, ax2=None, wg_list=wg_list,
                      y_range_real_spectrum=y_range_real_spectrum,
                      y_ticklabels_real_spectrum=y_ticklabels_real_spectrum,
                      projection=projection)
    get_imag_spectrum(ax1=ax2, ax2=None, wg_list=wg_list,
                      y_range_imag_spectrum=y_range_imag_spectrum,
                      y_ticklabels_imag_spectrum=y_ticklabels_imag_spectrum)

    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks([0, WGam.D.L/2, WGam.D.L])
        ax.set_xticklabels([r"0", r"L/2", r"L"])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    for n, ax in enumerate(f.get_axes()):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.tick_params(axis='both', which='minor', bottom='off',
                       left='off', right='off', top='off')

    f.text(0.28, -0., 'Spatial coordinate x', ha='center')
    f.text(0.79, -0., 'Spatial coordinate x', ha='center')
    f.text(-0.01, 0.99, 'a', weight='bold', size=12)
    f.text(0.5, 0.99, 'b', weight='bold', size=12)

    # plt.tight_layout(w_pad=1.5)
    plt.tight_layout(w_pad=1.75)
    # plt.tight_layout(w_pad=0.8, h_pad=0.2)
    plt.subplots_adjust(hspace=0.0, wspace=0.45)

    # plot_png(fig=f)
    # plt.show()

    plt.savefig(figname, bbox_inches='tight')


def plot_dynamics(wg_list, figname=None, y_range_trajectory=None,
                  y_axis_step_length=5):

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
                                               figsize=(6.2, 5.0/2.5*2.), dpi=220,
                                               sharex=False, sharey=False)
    get_real_spectrum_new(ax1=ax1, ax2=ax2, wg_list=wg_list, projection=True)
    get_trajectories_new(ax1=ax3, ax2=ax4, wg_list=wg_list,
                         y_range_trajectory=y_range_trajectory,
                         y_axis_step_length=y_axis_step_length)

    # for ax in (ax3, ax4):
    for ax in (ax1, ax2, ax3, ax4):
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
    # f.text(-0.01, 0.94, 'a', weight='bold', size=12)
    # f.text(-0.01, 0.49, 'b', weight='bold', size=12)
    f.text(-0.01, 1.00, 'a', weight='bold', size=12)
    f.text(-0.01, 0.50, 'b', weight='bold', size=12)

    plt.tight_layout(w_pad=0.8, h_pad=0.25)
    plt.subplots_adjust(hspace=0.2)

    # plot_png(fig=f)

    plt.savefig(figname, bbox_inches='tight')


def plot_uniform():
    wg_kwargs_am = {
        # pos. dep. loss configuration/ works for (approx.) uniform loss
        # (scaled with eps^2)
        # use N=2.05 to fit simulation setup
        'N': 2.05,
        'loop_type': 'Bell',
        'loop_direction': '-',
        'init_state': 'a',
        'init_state_method': 'energy',
        'W': 1,
        'L': 100,
        'eta':  0.6,
        'eta0': 0.0,
        'x_R0': 0.1,
        'y_R0': 0.85,
        'init_phase': 0.3,
        'switch_losses_on_off': True,
        # 'calc_adiabatic_state': True
    }
    # wg_kwargs_am = {
    #     # pos. dep. loss configuration/ works for (approx.) uniform loss
    #     # (scaled with eps^2)
    #     # use N=2.05 to fit simulation setup
    #     'N': 2.05,
    #     'loop_type': 'Bell',
    #     'loop_direction': '-',
    #     'init_state': 'a',
    #     'init_state_method': 'energy',
    #     'W': 0.5,
    #     'L': 100*0.5,
    #     'eta':  0.6/0.5,
    #     'eta0': 0.0,
    #     'x_R0': 0.1*0.5,
    #     'y_R0': 0.85/0.5,
    #     'init_phase': 0.3/0.5,
    #     'switch_losses_on_off': True,
    #     # 'calc_adiabatic_state': True
    # }

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
    wg_list = [DirichletReduced(**wg_kwargs) for wg_kwargs in wg_kwarg_list]
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

    plot_dynamics(wg_list,
                  figname="uniform_reduced_trajectory.pdf",
                  y_axis_step_length=10,
                  y_range_trajectory=[5e-34, 1e2])

    plot_spectrum(wg_list,
                  figname="uniform_reduced_spectrum.pdf",
                  y_range_real_spectrum=[-0.65, 1.2],
                  y_ticklabels_real_spectrum=5,
                  y_range_imag_spectrum=[-1.1, 0.1],
                  y_ticklabels_imag_spectrum=5)

    plot_parameter_trajectory(wg=WGam, figname="uniform_path.pdf")
    # plot_parameter_trajectory_p1_p2(wg=WGam, figname="uniform_path_p1_p2.pdf")


def plot_position_dependent():
    wg_kwargs_am = {
        # pos. dep. loss configuration
        # (scaled with eps^2)
        # use N=2.05 to fit simulation setup
        'N': 2.05,
        'loop_type': 'Bell',
        'loop_direction': '-',
        'init_state': 'a',
        'init_state_method': 'energy',
        'W': 1,
        'L': 100,
        'eta':  1.0,
        'eta0': 1.0, # here makes no difference if 0.0 or 1.0; only necessary for extension in (sigma, delta) plane?
        'x_R0': 0.1,
        'y_R0': 0.85,
        'init_phase': 0.0,
        'switch_losses_on_off': True,
        'sigma': 1e-3
    }

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
    wg_list = [DirichletPositionDependentLossReduced(**kw) for kw in wg_kwarg_list]
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

    plot_dynamics(wg_list,
                  figname="pos_dep_reduced_trajectory.pdf",
                  # y_axis_step_length=5,
                  y_axis_step_length=2,
                  # y_range_trajectory=[1e-11, 1e1])
                  y_range_trajectory=[1e-6, 1e1])

    plot_spectrum(wg_list,
                  figname="pos_dep_reduced_spectrum.pdf",
                  y_range_real_spectrum=[-1.1, 1.1],
                  y_ticklabels_real_spectrum=3,
                  # y_range_imag_spectrum=[-4.0, 0.2],
                  y_range_imag_spectrum=[-3.2, 0.2],
                  y_ticklabels_imag_spectrum=4)

    plot_parameter_trajectory(wg=WGam, figname="pos_dep_path.pdf")


def selector(pos_dep=False):
    if pos_dep:
        plot_position_dependent()
    else:
        plot_uniform()


if __name__ == '__main__':
    argh.dispatch_command(selector)
