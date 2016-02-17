#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import scipy

import argh

import ep.plot
from ep.waveguide import DirichletReduced


ep.plot.get_defaults()
colors, parula, parula_discrete = ep.plot.get_colors(N=10)


def plot_parameter_trajectory_p1_p2(W=1.0, remove_inside=False, show=False):
    wg_kwargs = dict(N=2.05,
                     L=25*W,
                     W=W,
                     x_R0=0.1*W,
                     y_R0=0.85/W,
                     init_phase=0.3/W,
                     eta=0.6/W,
                     switch_losses_on_off=True,
                     loop_type='Bell',
                     tN=50)
    # wg_kwargs = dict(N=2.5,
    #                  L=25*W,
    #                  W=W,
    #                  x_R0=0.25*W,
    #                  y_R0=0.5/W,
    #                  init_phase=-0.3/W,
    #                  eta=0.3/W,
    #                  switch_losses_on_off=True,
    #                  loop_type='Bell')
    WGam = DirichletReduced(**wg_kwargs)

    eps, delta = WGam.get_cycle_parameters()
    eps_EP, delta_EP = WGam.x_EP, WGam.y_EP

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6.2, 2.1), dpi=220)

    phi = lambda e, d: np.arctan2((d - WGam.init_phase)/WGam.y_R0, e/WGam.x_R0)
    R = lambda e, d: np.hypot(e/WGam.x_R0, (d - WGam.init_phase)/WGam.y_R0)

    get_p1 = lambda e, d: R(e, d)*np.sin(2*phi(e, d))
    get_p2 = lambda e, d: R(e, d)*np.cos(2*phi(e, d))

    # plt.plot(WGam.t, R(eps, delta))
    # plt.show()

    EPS, DELTA = np.meshgrid(eps, delta)
    delta_to_eps = lambda d: WGam.x_R0*0.5*(1. + np.cos(np.pi*(d - WGam.init_phase)/WGam.y_R0))
    mask = EPS < delta_to_eps(DELTA)
    EPS, DELTA = [X[mask] for X in (EPS, DELTA)]

    eps_im_branch_cut = np.linspace(eps_EP, eps.max()*1.5, 100)
    delta_im_branch_cut = 0.*eps_im_branch_cut
    eps_re_branch_cut = np.linspace(0.0, eps_EP, 100)
    delta_re_branch_cut = delta_im_branch_cut
    delta_horizontal_line = np.linspace(delta[0], delta[-1], 100)
    eps_const = 0.*delta_horizontal_line + 1e-2

    p1 = get_p1(eps, delta)
    p2 = get_p2(eps, delta)
    p1_EP = get_p1(eps_EP, delta_EP)
    p2_EP = get_p2(eps_EP, delta_EP)
    P1 = get_p1(EPS, DELTA)
    P2 = get_p2(EPS, DELTA)

    def from_p(p1, p2):
        eps_from_p = np.sqrt(p1**2 + p2**2 + p2*np.sqrt(p1**2 + p2**2))/np.sqrt(2.)
        delta_from_p = p1/2.*np.sqrt(p1**2 + p2**2)/eps_from_p
        delta_from_p = delta_from_p*WGam.y_R0 + WGam.init_phase
        eps_from_p *= WGam.x_R0
        return eps_from_p, delta_from_p

    def get_circles(pp1, pp2, p1_0=0.0, p2_0=0.0, radius=0.5):
        R_circle = radius*np.hypot(p2_0 - pp2, p1_0 - pp1)
        phi_circle = np.arctan2(-p2_0 + pp2, -p1_0 + pp1)

        p1_circle = p1_0 + R_circle*np.cos(phi_circle)
        p2_circle = p2_0 + R_circle*np.sin(phi_circle)
        eps_circle, delta_circle = from_p(p1_circle, p2_circle)
        return p1_circle, p2_circle, eps_circle, delta_circle

    # p1_circle, p2_circle, eps_circle, delta_circle = get_circles(p1, p2, radius=0.5)

    # ax1.plot(p1, p2, "r-")
    # ax1.plot(p1_circle, p2_circle, "b-")
    # ax1.plot(p1_EP, p2_EP, "ko")
    # ax2.plot(delta_circle, eps_circle, "r-")
    # ax2.plot(delta, eps, "b-")
    # ax2.plot(delta_EP, eps_EP, "ko")
    # plt.show()

    if remove_inside:
        xydata = np.dstack([p1, p2])
        xydata = xydata.reshape((-1, 2))
        curve = Path(xydata)
        P1_P2_flat = np.vstack([P1.flatten(), P2.flatten()]).T
        inside = curve.contains_points(P1_P2_flat, radius=0.0)
        P1, P2 = [Z[~inside] for Z in (P1, P2)]

    p1_im_branch_cut = get_p1(eps_im_branch_cut, delta_im_branch_cut)
    p2_im_branch_cut = get_p2(eps_im_branch_cut, delta_im_branch_cut)

    p1_re_branch_cut = get_p1(eps_re_branch_cut, delta_re_branch_cut)
    p2_re_branch_cut = get_p2(eps_re_branch_cut, delta_re_branch_cut)

    p1_line_2 = get_p1(eps_const, delta_horizontal_line)
    p2_line_2 = get_p2(eps_const, delta_horizontal_line)

    plot_kwargs = dict(lw=1.0, ms=0.5, mec='none', clip_on=False)
    # ax1.plot(DELTA, EPS, "o", color=colors[-1], **plot_kwargs)
    # ax2.plot(P1, P2, "o", color=colors[-1], **plot_kwargs)

    radii = np.linspace(1., 0.0, 6000)
    for n, r in enumerate(radii):
        # if r < 0.85:
        #     r = r*1.075
        #     step = 2
        #     p1_p, p2_p = [np.concatenate([u[::step], [u[0]]]) for u in (p1, p2)]
        #     tck, _ = scipy.interpolate.splprep([p1_p, p2_p], k=5)#,
        #     s = np.linspace(0.0, 1.0, 1000)
        #     p1_p, p2_p = scipy.interpolate.splev(s, tck)
        #     p1_0, p2_0 = p1_EP, p2_EP
        # else:
        #     p1_p, p2_p = p1, p2
        #     p1_0, p2_0 = 0., 0.
        p1_p, p2_p = p1, p2
        p1_0, p2_0 = p1_EP*(1.-r**3), p2_EP*(1.-r**3)
        (p1_circle, p2_circle,
         eps_circle, delta_circle) = get_circles(p1_p, p2_p,
                                                 p1_0=p1_0, p2_0=p2_0,
                                                 radius=r)
        n = n/(len(radii))
        ax1.plot(delta_circle, eps_circle, "-", color=parula(n), lw=0.1)
        ax2.plot(p1_circle, p2_circle, "-", color=parula(n), lw=0.1)

    ax1.plot(delta, eps, color="grey", **plot_kwargs)
    ax1.plot(delta[0], eps[0], "o", color="grey", ms=5.0, mec='none')
    ax1.plot(delta[-1], eps[-1], "o", color="grey", ms=5.0, mec='none')
    ax1.plot(delta_re_branch_cut, eps_re_branch_cut, "--", color="k", **plot_kwargs)
    ax1.plot(delta_im_branch_cut[::5], eps_im_branch_cut[::5], "o", color="k", ms=2.0, mec='none')
    # ax1.plot(delta_horizontal_line, eps_const, "-", color=colors[1], **plot_kwargs)
    ax1.plot(delta_EP, eps_EP, "o", color="w", ms=5, mec='k')
    ax1.annotate('EP', (0.075, 0.04), textcoords='data',
                 weight='bold', size=12, color='black')


    ax1.set_xlim(-0.7, 1.3)
    ax1.set_ylim(-0.005, WGam.x_R0 + 0.005)
    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$\sigma$")
    # ax1.locator_params(axis='x', nbins=4)
    # ax1.locator_params(axis='y', nbins=4)

    ax2.plot(p1, p2, color="grey", **plot_kwargs)
    ax2.plot(p1[0], p2[0], "o", color="grey", ms=5.0, mec='none')
    ax2.plot(p1[-1], p2[-1], "o", color="grey", ms=5.0, mec='none')
    ax2.plot(p1_re_branch_cut, p2_re_branch_cut, "--", color="k", **plot_kwargs)
    ax2.plot(p1_im_branch_cut[::10], p2_im_branch_cut[::10], "o", color="k", ms=2.0, mec='none')
    # ax2.plot(p1_line_2, p2_line_2, "-", color=colors[1], **plot_kwargs)
    ax2.plot(p1_EP, p2_EP, "o", color="w", ms=5, mec='k')
    ax2.annotate('EP', (-0.45, 0.04), textcoords='data',
                 weight='bold', size=12, color='black')

    ax2.set_xlim(-0.85, 0.85)
    ax2.set_ylim(-1.1, 1.05)
    ax2.set_xlabel(r"$p_1$")
    ax2.set_ylabel(r"$p_2$")
    ax2.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='y', nbins=4)

    for ax in (ax1, ax2):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_tick_params(direction='out')

    ax1.annotate('a', (1.1, 0.1), textcoords='data',
                 weight='bold', size=12, color='black')
    ax2.annotate('b', (0.7, 1.0), textcoords='data',
                 weight='bold', size=12, color='black')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig("SI_Fig2.png", bbox_inches='tight')


if __name__ == '__main__':
    argh.dispatch_command(plot_parameter_trajectory_p1_p2)
