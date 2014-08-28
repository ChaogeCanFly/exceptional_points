#!/usr/bin/env python2.7

import brewer2mpl as brew
from ep.waveguide import Waveguide
from ep.helpers import FileOperations, cmap_discretize
from ep.helpers import map_trajectory, set_scientific_axes
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
import numpy as np


def circle_EP(filename=None, write_profile=False, **kwargs):
    """Calculate trajectories around the EP."""

    f = FileOperations(filename)

    WG = Waveguide(**kwargs)

    ##
    # plot amplitudes b0(x), b1(x)
    ##
    ax0 = plt.subplot2grid((3,3), (0,0), colspan=2)
    ax0.set_label("x [a.u.]")
    ax0.set_xlabel(r"x [a.u.]")
    ax0.set_ylabel(r"Amplitudes $|b_n(x)|$")
    set_scientific_axes(ax0)

    x, b0, b1 = WG.solve_ODE()

    # get adiabatic predictions
    b0_ad, b1_ad = (WG.Psi_adiabatic[:,0],
                    WG.Psi_adiabatic[:,1])

    # account for initial population of states a and b
    b0_ad *= abs(b0[0])
    b1_ad *= abs(b1[0])

    ax0.semilogy(x, abs(b0), "r-", label=r"$|b_0|$")
    ax0.semilogy(x, abs(b1), "g-", label=r"$|b_1|$")
    ax0.semilogy(x, abs(b0_ad), "b--", label=r"$|b_0^{\mathrm{ad}}|$")
    ax0.semilogy(x, abs(b1_ad), "k--", label=r"$|b_1^{\mathrm{ad}}|$")

    f.write("")
    f.write("Diodicity D = {}".format(abs(b0[-1])/abs(b1[-1])))
    f.write("Diodicity D^-1 = {}".format(abs(b1[-1])/abs(b0[-1])))

    plt.rcParams.update({'font.size': 8})

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.,
               labelspacing=0.2,
               columnspacing=0.5,
               handlelength=3,
               handletextpad=0.2)

    ##
    # plot real/imag(E1(t)), imag(E2(t))
    ##
    ax1 = plt.subplot2grid((3,3), (1,0), colspan=2)
    ax2 = ax1.twinx()
    set_scientific_axes(ax1, axis='y')
    set_scientific_axes(ax2, axis='y')

    Ea, Eb = WG.eVals[:,0], WG.eVals[:,1]

    ax1.set_xlabel("x [a.u.]")
    ax1.set_ylabel("Energy")
    ax1.yaxis.set_label_position('left')
    ax1.plot(x, Ea.imag, "r-",label=r"$\mathrm{Im}(E_0$")
    ax1.plot(x, Eb.imag, "g-",label=r"$\mathrm{Im}(E_1)$")
    ax2.plot(x, Ea.real, "r--", label=r"$\mathrm{Re}(E_0)$")
    ax2.plot(x, Eb.real, "g--", label=r"$\mathrm{Re}(E_1)$")

    t_imag = map_trajectory(abs(b0), abs(b1),
                            Ea.imag, Eb.imag)
    t_real = map_trajectory(abs(b0), abs(b1),
                            Ea.real, Eb.real)
    ax1.plot(x, t_imag, "k-")
    ax2.plot(x, t_real, "k--")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2,
               labels1 + labels2,
               bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=4, mode="expand",
               borderaxespad=0.,
               labelspacing=0.2,
               columnspacing=0.5,
               handlelength=2.5,
               handletextpad=0.2)

    ##
    # plot wavefunction
    ##
    x0, y0 = WG.get_cycle_parameters(0.)
    x1, y1 = WG.get_cycle_parameters(WG.t)
    ####
    ####ax3 = subplot2grid((3,3), (2,0), colspan=2)
    ####WG.draw_wavefunction()
    ####WG.draw_dissipation_coefficient()
    ####WG.draw_boundary()
    ####tick_params(labelleft='off', left='off', right='off')
    ####ax3.set_xlabel("x [a.u.]")
    ####ax3.set_frame_on(False)
    #ax3.xaxis.set_ticklabels([])
    #ax3.get_xaxis().tick_bottom()

    ##
    # plot path around EP
    ##
    ax4 = plt.subplot2grid((3,3), (2,2))
    set_scientific_axes(ax4, axis='both')
    ax4.xaxis.set_label_position('top')
    ax4.yaxis.set_label_position('right')
    plt.xticks(rotation=30)

    ax4.set_xlabel(r"$\epsilon$", fontsize=10)
    ax4.set_ylabel(r"$\delta$", fontsize=10)
    dx = dy = 0.5e-2
    #ax4.set_xlim(min(x1)-dx, max(x1)+dx)
    #ax4.set_ylim(min(y1)-dy, max(y1)+dy)

    offset = WG.tN/5.
    dx1 = np.diff(x1)
    dy1 = np.diff(y1)
    plt.plot(x1[:], y1[:], "grey", ls="dotted")
    plt.plot(x0, y0, "ro", mew=0.0, mec='r')
    plt.plot([WG.x_EP,-WG.x_EP],
             [WG.y_EP, WG.y_EP],"ko")

    plt.quiver(x1[offset:-1:offset], y1[offset:-1:offset],
               dx1[offset::offset], dy1[offset::offset],
               units='xy', angles='xy', headwidth=6.,
               color='k', zorder=10)
    #ylim(-0.1,0.1)
    #text(WG.x_EP - WG.x_R0*0.2,
    #     WG.y_EP + WG.y_R0*4, "EP")

    ##
    # save figure
    ##
    #tight_layout()
    plt.subplots_adjust(#left=0.125,
                        bottom=None,
                        #right=0.9,
                        top=None,
                        wspace=0.5,
                        hspace=0.6)

    filename = "{0}_{1}".format(filename, kwargs.get('loop_direction'))
    f.write(filename + ".cfg")
    plt.savefig(filename + ".pdf")
    if write_profile:
        xi_lower, _ = WG.get_boundary()
        np.savetxt(filename + ".profile", zip(WG.t, xi_lower))    
    clf()


class Diodicity:
    """
    Sample the parameterspace (eta, L) to find maxima and minima
    of the diodicity D = abs(a/b).
    """

    def __init__(self, filename=None, d_eta=0.005, d_L=5, **kwargs):
        self.d_eta = d_eta
        self.d_L = d_L
        self.kwargs = kwargs
        ##self.eta = np.arange(0.0025, 0.06, d_eta)
        ##self.L = np.arange(82.5, 130, d_L)
        #self.eta = np.arange(0.005, 0.06, d_eta)
        #self.L = np.arange(80, 130, d_L)
        self.eta = np.arange(0.05, 0.6, 0.05)
        self.L = np.arange(100,340,20)
        #self.L = np.linspace(100, 320, self.N_L)
        #self.L = np.arange(50, 100, self.d_L)

        if filename:
            self.filename = filename
            self.f = FileOperations(self.filename)
            self.f.write("")
            self.f.write(("# {:>10}{:>10}{:>20}"
                          "{:>20}{:>20}{:>20}"
                          "{:>20}{:>20}").format("eta", "L",
                                                 "R0", "R0_b0", "R0_b1",
                                                 "R1", "R1_b0", "R1_b1"))
            self.f.write("#" + 141*"-")

    def get_flip_error(self, eta, L):
        """Return the flip-error R = abs(a/b)."""
        self.kwargs['eta'] = eta
        self.kwargs['L'] = L
        
        # flip-error R0
        self.kwargs['loop_direction'] = '-'
        WG = Waveguide(**self.kwargs)
        _, b0, b1 = WG.solve_ODE()
        R0_b0 = abs(b0[-1])
        R0_b1 = abs(b1[-1])
        R0 = abs(b0[-1]/b1[-1])

        # flip-error R1
        self.kwargs['loop_direction'] = '+'
        WG = Waveguide(**self.kwargs)
        _, b0, b1 = WG.solve_ODE()
        R1_b0 = abs(b0[-1])
        R1_b1 = abs(b1[-1])
        R1 = abs(b0[-1]/b1[-1])
        
        self.f.write(("{:>12}{:>10}{:>20}{:>20}"
                      "{:>20}{:>20}{:>20}{:>20}").format(eta, L,
                                                         R0, R0_b0, R0_b1,
                                                         R1, R1_b0, R1_b1))
        return R0, R1

    def get_array(self):
        """Return flip-errors on a grid.
            
            Parameters:
                N_eta: int
                N_L: int
                
            Returns:
                X, Y: (N,N) ndarray
                Z: (N,N,2) ndarray
        """
        X, Y = np.meshgrid(self.eta, self.L)

        Z = np.zeros_like(X)
        Z = np.dstack((Z, Z))
        
        for n, e in enumerate(self.eta):
            for m, l in enumerate(self.L):
                # incorporate proper index n <-> m
                Z[m,n,:] = self.get_flip_error(e, l)

        return X, Y, Z


    #def save_data(self):
    #    """Save data to textfile with current timestamp and used parameters."""
    #
    #    self.X, self.Y, self.Z = self.get_array()
    #    np.savetxt(self.filename + ".dat",
    #               np.hstack((self.X, self.Y, self.Z[...,0], self.Z[...,1])))


    def plot(self, external_file=None, save_eps=True):
        """Plot a heatmap of the parameterspace (eta, L) to find
        maxima and minima of the flip-error R = abs(a/b) for both
        clockwise and anticlockwise loop directions.
        """

        if external_file:
            filename = str(external_file).replace(".cfg", ".eps")
            idx = np.array([0,1,2,5])
            file = np.loadtxt(external_file, unpack=True)
            X, Y, Z0, Z1 = file[idx]

            self.eta, self.L = np.unique(X), np.unique(Y)
            lx, ly = len(self.eta), len(self.L)
            
            self.X, self.Y, Z0, Z1 = map(lambda x: x.reshape((lx,ly)),
                                         (X, Y, Z0, Z1))
            self.Z = np.dstack((Z0,Z1))

        # set global fontsize to 10
        plt.rcParams.update({'font.size': 10,
                             'axes.titlesize': 10})

        f, axes = plt.subplots(nrows=1, ncols=2)
        plt.subplots_adjust(wspace=0.25)

        loop_directions = ('counterclockwise', 'clockwise')
        for n, (ax, d) in enumerate(zip(axes, loop_directions)):

            p = ax.pcolormesh(self.X, self.Y, self.Z[:,:,n],
                              norm=LogNorm(1e-3,1e3))
                              #norm=LogNorm(1e-4,1e4))
            plt.xticks(rotation=45)

            # adjust colormap
            ls = p.norm([5e-2,2e-1,5e-1,2e0,5e0,2e1])
            bmap = brew.get_map('YlGnBu',
            #bmap = brew.get_map('Greys',
                                #'sequential', 7, reverse=True).mpl_colormap
                                'sequential', 9, reverse=True).mpl_colormap
            #bmap = cmap_discretize(bmap, ls)
            bmap.set_over('w')
            bmap.set_under('k')
            p.cmap = bmap

            # general axes properties
            ax.set_title('%s' % d, fontsize=10)
            ax.grid(which='minor')

            # x-axis
            ax.set_xlabel(r'$\eta$ (dissipation coefficient)')
            xoffset = np.diff(self.eta).mean()/2
            ax.set_xticks(self.eta + xoffset)
            ax.set_xticklabels(self.eta)

            fmt = plt.FuncFormatter(lambda x, p: '{:.2f}'.format(x-xoffset))
            ax.xaxis.set_major_formatter(fmt)
            #for label in ax.get_xticklabels()[1::2]:
            #    label.set_visible(False)
            ax.xaxis.set_minor_locator(FixedLocator(self.eta))

            # y-axis
            ax.set_ylabel(r'$L$ (system length)') 
            yoffset = np.diff(self.L).mean()/2
            ax.set_yticks(self.L + yoffset)
            ax.set_yticklabels(map(int, self.L))
            ax.yaxis.set_minor_locator(FixedLocator(self.L))

            # colorbar
            
            
            cb = f.colorbar(p, ax=ax, aspect=10,
                            orientation='horizontal')
            cb.set_label(r'$|b_0(L)|/|b_1(L)|$')

            #for clabel in cb.ax.get_xticklabels():
            #    clabel.set_rotation(30)
            #major_ticks = p.norm([2e-2,1e-1,1e0,1e1,5e1])
            #major_ticks = p.norm([1e-2,1e-1,1e0,1e1,1e2])
            #minor_ticks = p.norm([5e-2,7e-2,2e-1,5e-1,7e-1,2,5,7,20,50])
            #cb.ax.xaxis.set_ticks(major_ticks)
            #cb.ax.xaxis.set_ticklabels([r'$<\,2\cdot10^{{-2}}$',
                                        #r'$10^{{-1}}$', r'$10^0$',
                                        #r'$10^1$', r'$>5\cdot10^1$'])
            #cb.ax.xaxis.set_ticklabels([r'$10^{{-2}}$',
            #                            r'$10^{{-1}}$', r'$10^0$',
            #                            r'$10^1$', r'$10^2$'])
            #cb.ax.xaxis.set_ticks(minor_ticks, minor=True)
            ax.autoscale_view(True)

        if save_eps is True:
            #savefig("{}.eps".format(filename))
            plt.savefig("{}".format(filename))
        else:
            plt.show()


class ParseArguments:
    """Wrapper for the argparse module."""
    def __init__(self):
        import argparse as ap
        parser = ap.ArgumentParser(
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)

        p_major = parser.add_argument_group('Major options')
        p_major.add_argument("-N", default=1.01, type=float,
                            help="Number of open modes")
        p_major.add_argument("-l", "--length", default=100, type=int,
                            help="System length")
        p_major.add_argument("-e", "--eta", default=0.03, type=float,
                            help="Dissipation coefficient value")

        p_minor = parser.add_argument_group('Minor options')
        p_minor.add_argument("-i", "--init-state", default="c", type=str,
                            help="Initial state")
        p_minor.add_argument("-p", "--init-phase", default=0.0, type=float,
                            help="Initial phase (in multiples of pi)")
        p_minor.add_argument("-c", "--cycletype", default="Circle", type=str,
                            help="Cycle type" )
        p_minor.add_argument("-d", "--direction", default="-", type=str,
                            help="Loop direction")
        p_minor.add_argument("-t", "--theta", default=0.0, type=float,
                            help="Phase difference between boundaries")
        p_minor.add_argument("-a", "--adiabatic", action="store_true",
                            help="Calculate adiabatic solution")
        p_minor.add_argument("-b", "--boundary", action="store_true",
                            help="Whether to write profile boundary")

        calc = parser.add_mutually_exclusive_group(required=True)
        calc.add_argument("--loop", action="store_true",
                           help="Loop the EP and prepare plots")
        calc.add_argument("--heatmap", action="store_true",
                           help="Get diodicity heatmap")
        calc.add_argument("--riemann", action="store_true",
                           help="Plot Riemann sheet structure")

        self.args = parser.parse_args()
        self.filename = self.get_filename()
        self.f = FileOperations(self.filename)

    def get_filename(self):
        args = self.args.__dict__
        #p0 = args.get('init_phase')
        filename = ("N_{N}_{cycletype}_phase_{init_phase:.3f}pi"
                    "_init_state_{init_state}").format(**args)

        if not args.get('heatmap'):
            # add length and eta for --riemann and --loop
            filename = filename + ("_L_{length}_eta_{eta}").format(**args)
        return filename.replace(".","")


    def print_values(self):
        """Print all variables defined via argparse."""

        args = self.args.__dict__
        major, minor, calc = [ m.copy() for m in 3*(args, ) ]
                                
        m = ['N', 'length', 'eta']
        c = ['loop', 'heatmap', 'riemann']
        
        for key in m + c:
            del minor[key]
        o = [ k for (k,v) in minor.items() ]
        for key in o + c:
            del major[key]
        for key in m + o:
            del calc[key]

        self.f.write(50*"#" + "\n#")
        self.f.write("#  Input variables:")
        self.f.write("#  ----------------")
        for dicts in major, calc, minor:
            self.f.write("#")
            for key in dicts:
                self.f.write("#  {:<12} {:<12}".format(key, dicts[key]))
        self.f.write("#\n" + 50*"#" + "\n")
        self.f.write("#  Warning: is normalization symmetric?")
        self.f.close()


if __name__ == '__main__':

    P = ParseArguments()
    args = P.args
    P.print_values()

    params = {
        'L': args.length,
        'N': args.N,
        'eta': args.eta,
        'init_state': args.init_state,
        'init_phase': args.init_phase*pi,
        'loop_type': args.cycletype,
        'loop_direction': args.direction,
        'theta': args.theta,
        'calc_adiabatic_state': args.adiabatic
    }

    if args.loop:
        for d in '+', '-':
            params['loop_direction'] = d
            circle_EP(filename=P.get_filename(),
                      write_profile=args.boundary, **params)
    elif args.heatmap:
        D = Diodicity(filename=P.get_filename(), **params)
        D.get_array()
        #D.save_data()
        #D.plot(save_eps=True)
    elif args.riemann:
        plot_riemann_sheets(filename=P.get_filename(), **params)
