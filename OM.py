#!/usr/bin/env python

import matplotlib
#matplotlib.use('Agg')

from EP_OptoMech import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter, MultipleLocator, FixedLocator
import brewer2mpl as brew

def plot_riemann_sheets(**kwargs):
    """Plot local Riemann sheet structure of the OM Hamiltonian."""
    # init EP_OptoMech object 
    OM = EP_OptoMech(**kwargs)
    # sample H eigenvalues
    OM.solve_ODE()
    X, Y, Z = OM.sample_H()
    x, y = OM.get_cycle_parameters(OM.t)

    # plot 3D surface
    fig = figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    
    axes =  ax1, ax2
    parts = imag, real
    
    for ax, part in zip(axes, parts):
        # plot both eigenvalues
        for n in (0, 1):
            ax.plot_surface(X, Y, part(Z[:,:,n]),
                            cmap=cm.jet, linewidth=0.1)
            
            Ea = part(OM.eVals[:,0])
            Eb = part(OM.eVals[:,1])

            ax.plot(x, y, Ea, "r-")
            ax.plot(x, y, Eb, "g-")
    show()    


def circle_EP(**kwargs):
    """Calculate trajectories around the EP.
    
        Parameters:
            L:  float, optional
                System length.
                
        Returns:
            None
    """
    OM = EP_OptoMech(**kwargs)
    
    ##
    # plot amplitudes b0(x), b1(x)
    ##
    subplot2grid((2,3), (0,0), colspan=2)
    
    x, b0, b1 = OM.solve_ODE()
    
    # get adiabatic predictions
    b0_ad, b1_ad = (OM.Psi_adiabatic[:,0],
                    OM.Psi_adiabatic[:,1])
    
    # account for initial population of states a and b
    b0_ad *= abs(b0[0])
    b1_ad *= abs(b1[0])

    # plot data
    semilogy(x, abs(b0), "r-", label=r"$|b_0|$")
    semilogy(x, abs(b1), "g-", label=r"$|b_1|$")
    semilogy(x, abs(b0_ad), "b--", label=r"$|b_0^{\mathrm{ad}}|$")
    semilogy(x, abs(b1_ad), "k--", label=r"$|b_1^{\mathrm{ad}}|$")
    #np.savetxt('data.dat', zip(x, abs(b0), abs(b1), abs(b0_ad)))
    xlim(0, OM.T)
    ymin, ymax = ylim()
    ylim(1e-7, 1e9)
    
    xlabel("t [a.u.]")
    ylabel(r"Amplitudes $|b_n(x)|$")
    
    print "Diodicity D =", abs(b0[-1])/abs(b1[-1])
    
    params = {'legend.fontsize': 'small',
              'size': 'xsmall'}
    rcParams.update(params)
    
    legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.,
           labelspacing=0.2,
           columnspacing=0.5,
           handlelength=3,
           handletextpad=0.2)
    
    ##
    # plot real/imag(E1(t)), imag(E2(t))
    ##
    ax1 = subplot2grid((2,3), (1,0), colspan=2)
    ax2 = ax1.twinx()
    set_scientific_axes(ax1, axis='y')
    set_scientific_axes(ax2, axis='y')
    
    Ea, Eb = OM.eVals[:,0], OM.eVals[:,1]
    t_imag = map_trajectory(abs(b0), abs(b1),
                             imag(Ea), imag(Eb))
    t_real = map_trajectory(abs(b0), abs(b1),
                             real(Ea), real(Eb))

    ax1.plot(x, imag(Ea), "r-", label=r"$\mathrm{Im} E_0$")
    ax1.plot(x, imag(Eb), "g-", label=r"$\mathrm{Im} E_1$") 
    ax1.plot(x, t_imag, "k-", label=r'$\mathrm{Im}$ $\mathrm{projection}$')
    ax2.plot(x, real(Ea), "r--", label=r"$\mathrm{Re} E_0$")
    ax2.plot(x, real(Eb), "g--", label=r"$\mathrm{Re} E_1$")
    ax2.plot(x, t_real, "k--", label=r'$\mathrm{Re}$ $\mathrm{projection}$')
    
    ax1.set_xlabel("t [a.u.]")
    ax1.set_ylabel("Energy")
    ax1.yaxis.set_label_position('left')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2,
               labels1 + labels2,
               bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2, mode="expand",
               borderaxespad=0.,
               labelspacing=0.2,
               columnspacing=0.5,
               handlelength=2.5,
               handletextpad=0.2
               )
    
    ##
    # plot path around EP
    ##
    ax = subplot2grid((2,3), (0,2), aspect=1)
    set_scientific_axes(ax, axis='both')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')
    
    xlabel(r"$\epsilon$")
    ylabel(r"$\delta$")
    plot([real(OM.B1),imag(OM.B1)],
         [real(OM.B2),imag(OM.B2)],"ko-")
    
    dx = dy = OM.R * 0.2
    x0, y0 = OM.get_cycle_parameters(0.)
    x1, y1 = OM.get_cycle_parameters(OM.t)
    xlim(min(x1)-dx,max(x1)+dx)
    ylim(min(y1)-dy,max(y1)+dy)
    
    plot(x1, y1, "grey", ls="dotted")
    plot(x0, y0, "ro", mew=0.0, mec='r')
    offset=OM.tN/3.
    start=offset/4.
    quiver(x1[start:-1:offset], y1[start:-1:offset],
           x1[1+start::offset]-x1[start:-1:offset],
           y1[1+start::offset]-y1[start:-1:offset],
           units='xy', angles='xy', headwidth=6.,
           color='k', zorder=10)
           #width=2e-2,
           #scale_units='inches',
           #scale=0.0015, color='k', zorder=1)
    tick_params(axis='both', which='major', labelsize=8)
    text(real(OM.B1) - OM.R*0.15,
         imag(OM.B1) + OM.R*0.2, "EP")
    
    ##
    # save figure
    ##
    subplots_adjust(#left=0.125,
                    bottom=None,
                    #right=0.9,
                    top=None,
                    wspace=0.6,
                    hspace=0.6)
    
    filename = ("R_{R}_T_{T}_phase_{init_loop_phase}_"
                "state_{init_state}_{loop_direction}").format(**kwargs)
    print "writing ", filename
    savefig(filename + ".pdf")
    np.savetxt(filename + ".dat", zip(OM.Psi[:,0], OM.Psi[:,1], b0, b1, Ea, Eb,
                                      OM.eVecs_r[:,0,0], OM.eVecs_r[:,1,0],
                                      OM.eVecs_r[:,0,1], OM.eVecs_r[:,1,1]))
    clf()


if __name__ == '__main__':
    
    params = {
        'T': 32,
        #'R': 0.0625,
        'R': 0.25,
        'init_loop_phase': 3.70031,                # Im(lambda)=0, R=0.25
        #'init_loop_phase': 3.86493,                # Im(lambda)=0, R=0.0625
        #'init_loop_phase': 1.01208,                # Re(lambda)=0, R=0.25
        #'init_loop_phase': 0.847457,               # Re(lambda)=0, R=0.0625
        'calc_adiabatic_state': True
    }
    
    print "Warning: is normalization symmetric?"
    
    if 1:
        for d in '+', : #'+', '-':
            for s in 'a', : #'a', 'b':
                params['loop_direction'] = d
                params['init_state'] = s
                circle_EP(**params)
    else:
        plot_riemann_sheets(**params)
