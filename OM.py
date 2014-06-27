#!/usr/bin/env python

import matplotlib
#matplotlib.use('Agg')

from EP_OptoMech import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import LogFormatter, MultipleLocator, FixedLocator
import brewer2mpl as brew

    
def plot_riemann_sheets(part=np.imag, scale=6.5, wireframe_skip=5.,
                        xN=153, yN=152, **kwargs):
    """Plot local Riemann sheet structure of the OM Hamiltonian."""

    import mayavi.mlab as mlab
    from EP_Helpers import map_trajectory, get_height_profile
    import matplotlib.pyplot as plt
    
    #part = np.imag
    part = np.real
    
    OM = EP_OptoMech(**kwargs)
    
    x, y = OM.get_cycle_parameters(OM.t)
    _, c1, c2 = OM.solve_ODE()
    
    e1 = part(OM.eVals[:,0])
    e2 = part(OM.eVals[:,1])
    z = map_trajectory(c1, c2, e1, e2)
    z = e1    
    
    X, Y, Z = OM.sample_H(xN=xN, yN=yN)
    E1 = part(Z[...,1])
    E0 = part(Z[...,0])
    
    nx = np.sqrt(len(E1.ravel())).astype(int)/2
    ny = nx
    
    def get_custom_cmap():
        red = tuple(map(lambda x: x/255., (228,26,28)))
        blue = tuple(map(lambda x: x/255., (55,126,184)))
        cmap = LinearSegmentedColormap.from_list('RdBu_custom',
                                                 [red, blue], N=256)
        return cmap(np.arange(256))*255.
    
    RdBu_custom = get_custom_cmap()
    

    fig = mlab.figure(size=(1400,1000), bgcolor=(1,1,1))
    
    line_color = (0.25, 0.25, 0.25)
    mlab.plot3d(x, y, z/scale,
                color=line_color,
                opacity=1.,
                tube_radius=0.001)
    
    W_Gauss_Fermi, W_Fermi, wmax, wmin = get_height_profile(X, Y, rho_y=1e-2, 
                                                            sigma_x=1e-4)
    
    
    if part is np.real:
        
        E1s1p1 = mlab.mesh(X[:nx+1,:ny+1],
                         Y[:nx+1,:ny+1],
                         E1[:nx+1,:ny+1]/scale,
                         scalars=W_Gauss_Fermi[:nx+1,:ny+1],
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)        
        E1s2 = mlab.mesh(X[:nx+1,ny:],
                         Y[:nx+1,ny:],
                         E1[:nx+1,ny:]/scale,
                         scalars=W_Fermi[:nx+1,ny:],         
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)
        
        E0s1 = mlab.mesh(X[...,:ny+1],
                         Y[...,:ny+1],
                         E0[...,:ny+1]/scale,
                         scalars=W_Gauss_Fermi[...,:ny+1],
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)
        E0s2 = mlab.mesh(X[:nx+1,ny:],
                         Y[:nx+1,ny:],
                         E0[:nx+1,ny:]/scale,
                         scalars=W_Fermi[:nx+1,ny:],         
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)
        E0s3 = mlab.mesh(X[nx+1:,ny:],
                         Y[nx+1:,ny:],
                         E0[nx+1:,ny:]/scale,
                         scalars=W_Fermi[nx+1:,ny:],
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)
        
        E0s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
        E0s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
        E0s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        
        mlab.surf(X[nx+1::wireframe_skip,::wireframe_skip],
                  Y[nx+1::wireframe_skip,::wireframe_skip],
                  E0[nx+1::wireframe_skip,::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0,0,0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)
        
        mlab.surf(X[:nx+2:wireframe_skip,::wireframe_skip],
                  Y[:nx+2:wireframe_skip,::wireframe_skip],
                  E0[:nx+2:wireframe_skip,::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0,0,0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)        
        mesh_args = {
            
        }
        
        #E1s1 = mlab.mesh(X[...,:ny+1],
        #                 Y[...,:ny+1],
        #                 E1[...,:ny+1]/scale,
        #                 scalars=W_Gauss_Fermi[...,:ny+1],
        #                 opacity=0.9,
        #                 vmin=wmin, vmax=wmax)

        E1s1p2 = mlab.mesh(X[nx+1:,:ny+1],
                         Y[nx+1:,:ny+1],
                         E1[nx+1:,:ny+1]/scale,
                         scalars=W_Gauss_Fermi[nx+1:,:ny+1],
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)
        E1s3 = mlab.mesh(X[nx+1:,ny:],
                         Y[nx+1:,ny:],
                         E1[nx+1:,ny:]/scale,
                         scalars=W_Fermi[nx+1:,ny:],
                         opacity=0.9,
                         vmin=wmin, vmax=wmax)
        #E1s2 = mlab.mesh(X[:nx+1,ny:],
        #                 Y[:nx+1,ny:],
        #                 E1[:nx+1,ny:]/scale,
        #                 scalars=W_Fermi[:nx+1,ny:],         
        #                 opacity=0.9,
        #                 vmin=wmin, vmax=wmax)

        
        #E1s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        E1s1p1.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        E1s1p2.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        E1s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        E1s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]

        mlab.surf(X[nx+1::wireframe_skip,::wireframe_skip],
                  Y[nx+1::wireframe_skip,::wireframe_skip],
                  E1[nx+1::wireframe_skip,::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0,0,0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)
        
        mlab.surf(X[:nx+2:wireframe_skip,::wireframe_skip],
                  Y[:nx+2:wireframe_skip,::wireframe_skip],
                  E1[:nx+2:wireframe_skip,::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0,0,0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)
        
        ##
        
        #E0s1 = mlab.mesh(X[...,:ny+1],
        #                 Y[...,:ny+1],
        #                 E0[...,:ny+1]/scale,
        #                 scalars=W_Gauss_Fermi[...,:ny+1],
        #                 opacity=0.75,
        #                 vmin=wmin, vmax=wmax)
        #E0s2 = mlab.mesh(X[:nx+1,ny:],
        #                 Y[:nx+1,ny:],
        #                 E0[:nx+1,ny:]/scale,
        #                 scalars=W_Fermi[:nx+1,ny:],         
        #                 opacity=0.75,
        #                 vmin=wmin, vmax=wmax)
        #E0s3 = mlab.mesh(X[nx+1:,ny:],
        #                 Y[nx+1:,ny:],
        #                 E0[nx+1:,ny:]/scale,
        #                 scalars=W_Fermi[nx+1:,ny:],
        #                 opacity=0.75,
        #                 vmin=wmin, vmax=wmax)
        #
        #E0s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
        #E0s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
        #E0s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        #
        #mlab.surf(X[nx+1::wireframe_skip,::wireframe_skip],
        #          Y[nx+1::wireframe_skip,::wireframe_skip],
        #          E0[nx+1::wireframe_skip,::wireframe_skip]/scale,
        #          opacity=0.05,
        #          color=(0,0,0),
        #          representation='wireframe',
        #          vmin=wmin, vmax=wmax)
        #
        #mlab.surf(X[:nx+2:wireframe_skip,::wireframe_skip],
        #          Y[:nx+2:wireframe_skip,::wireframe_skip],
        #          E0[:nx+2:wireframe_skip,::wireframe_skip]/scale,
        #          opacity=0.05,
        #          color=(0,0,0),
        #          representation='wireframe',
        #          vmin=wmin, vmax=wmax)
 
    else:
        s2 = mlab.mesh(X, Y, E0/scale,
                        scalars=W_Fermi, 
                        representation='surface',
                        opacity=0.8,
                        vmin=-wmin, vmax=wmax)        
        s1 = mlab.mesh(X, Y, E1/scale,
                        scalars=W_Fermi, 
                        representation='surface',
                        opacity=0.8, 
                        vmin=-wmin, vmax=wmax)
        
        s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
        s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom
        
        mlab.surf(X[::wireframe_skip,::wireframe_skip],
                  Y[::wireframe_skip,::wireframe_skip],
                  E1[::wireframe_skip,::wireframe_skip]/scale,
                  representation='wireframe',
                  opacity=0.05, 
                  color=(0,0,0),
                  vmin=-wmin, vmax=wmax)
        
        mlab.surf(X[::wireframe_skip,::wireframe_skip],
                  Y[::wireframe_skip,::wireframe_skip],
                  E0[::wireframe_skip,::wireframe_skip]/scale,
                  representation='wireframe',
                  opacity=0.05,
                  color=(0,0,0),
                  vmin=-wmin, vmax=wmax)
    
    for E in E0, E1:
        for xx, yy, zz in ([X[:,0],  Y[:,0],  E[:,0]/scale],
                           [X[:,-1], Y[:,-1], E[:,-1]/scale],
                           [X[0,:],  Y[0,:],  E[0,:]/scale],
                           [X[-1,:], Y[-1,:], E[-1,:]/scale]):
            
            edge_color = (0.25, 0.25, 0.25)
            
            n = np.where(abs(np.diff(zz)) >= 0.01)[0]
            if n.size:
                n = n[0]
                mlab.plot3d(xx[:n+1], yy[:n+1], zz[:n+1],
                            color=edge_color, 
                            tube_radius=0.0005)
                mlab.plot3d(xx[n+1:], yy[n+1:], zz[n+1:],
                            color=edge_color, 
                            tube_radius=0.0005)
            else:
                mlab.plot3d(xx, yy, zz,
                            color=edge_color, 
                            tube_radius=0.0005)
    
    
    mlab.points3d(x[0], y[0], z[0]/scale,
                  color=line_color,
                  scale_factor=0.0075,
                  mode='sphere')
    
    u, v, w = [ np.gradient(n) for n in x, y, z/scale ]
    
    if part is np.real:
        x, y, z, u, v, w = [ n[-1] for n in x, y, z/scale, u, v, w ]
        mlab.quiver3d(x, y, z, u, v, w,
                    color=line_color,
                    scale_factor=200,
                    resolution=200,
                    mode='cone'        
                    )
    else:
        x, y, z, u, v, w = [ n[-1] for n in x, y, z/scale, u, v, w ]
        mlab.quiver3d(x, y, z, u, v, w,
                    color=line_color,
                    scale_factor=200,
                    resolution=50,
                    mode='cone'
                    )
        
    #mlab.outline(extent=ext,
    #             line_width=2.5,
    #             color=(0.5, 0.5, 0.5))
    
    mlab.axes(figure=fig,
              color=(0.25,0.25,0.25),
              extent=[X.min(),X.max(),
                      Y.min()*0.95,Y.max()*0.95,
                      E0.min()/3.5,E1.max()/3.5])
    
    mlab.view(azimuth=40, elevation=55)
    #mlab.view(focalpoint = [0.0, 1.0, 0.0])
    mlab.draw()
    fig.scene.render_window.aa_frames = 8
    mlab.show()
    


def circle_EP(**kwargs):
    """Calculate trajectories around the EP.
    
        Parameters:
        -----------
            L:  float, optional
                System length.
                
        Returns:
        --------
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
        'T': 5,
        #'R': 0.0625,
        #'R': 0.5,
        #'init_loop_phase': 3.70031,                # Im(lambda)=0, R=0.25
        #'init_loop_phase': 3.86493,                # Im(lambda)=0, R=0.0625
        #'init_loop_phase': 1.01208,                # Re(lambda)=0, R=0.25
        #'init_loop_phase': 0.847457,               # Re(lambda)=0, R=0.0625
        #'calc_adiabatic_state': True
    }
    
    print "Warning: is normalization symmetric?"
    
    if 0:
        for d in '+', : #'+', '-':
            for s in 'a', : #'a', 'b':
                params['loop_direction'] = d
                params['init_state'] = s
                circle_EP(**params)
    else:
        params = {
                "T": 10., 
                "R": 1/20., 
                "gamma": 2., 
                "init_state": 'a', 
                "init_loop_phase": 1*pi/2., #1*pi*0, 
                "loop_direction": '-',
                "calc_adiabatic_state": False
                }
        
        plot_riemann_sheets(**params)
