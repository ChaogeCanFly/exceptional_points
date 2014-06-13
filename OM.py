#!/usr/bin/env python

import matplotlib
#matplotlib.use('Agg')

from EP_OptoMech import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import LogFormatter, MultipleLocator, FixedLocator
import brewer2mpl as brew

    
def plot_riemann_sheets(**kwargs):
    """Plot local Riemann sheet structure of the OM Hamiltonian."""

    import mayavi.mlab as mlab
    from EP_Helpers import map_trajectory
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    
    #cdict = {'red': [228,26,28],
             #'blue': [55,126,184]}
    #cmap_rb = LinearSegmentedColormap('cmap_rb', cdict,)
    #bmap = brew.get_map('Spectral', 'Diverging', 5)
    #cmap = bmap.mpl_colormap
    cmap = 'Spectral'
    cmap = 'Blues'
    
    part = np.imag
    #part = np.real
    
    OM = EP_OptoMech(**kwargs)
    x, y = OM.get_cycle_parameters(OM.t)
    _, c1, c2 = OM.solve_ODE()
    X, Y, Z = OM.sample_H(xN=51, yN=51)
    
    xx, yy = [ n.ravel() for n in X, Y ]
    zz = part(Z[...,0].ravel())
    
    
    #from scipy.spatial import Delaunay
    
    #points = zip(*(n.flat for n in (X, Y, part(Z[...,0]))))
    #print points
    #triangles = Delaunay(points).vertices
    #print triangles
    
    fig = mlab.figure(size=(1400,1000), bgcolor=(1,1,1))
    #mlab.triangular_mesh(xx,yy,zz,triangles[...,1:],
    #                     vmin=-1,vmax=1)
    
    if part is np.real:
        ext = [np.min(X), np.max(X),    
               np.min(Y), np.max(Y),
               -1, 1]
               #np.min(part(Z[...,0])), np.max(part(Z[...,0])))
    else:
        ext = [np.min(X), np.max(X),    
               np.min(Y), np.max(Y),    
               np.min(part(Z[...,0])), np.max(part(Z[...,1]))]
    
    


    E1 = part(Z[...,1])
    E0 = part(Z[...,0])
    
    nx = np.sqrt(len(E1.ravel())).astype(int)/2
    ny = nx
    
    red = tuple(map(lambda x: x/255., (228,26,28)))
    blue = tuple(map(lambda x: x/255., (55,126,184)))

    cmap = LinearSegmentedColormap.from_list('RdBu_custom',
                                             [red,red,red,red,
                                             #'white',
                                             blue,blue,blue,blue], N=256)
    RdBu_custom = cmap(np.arange(256))*255.
            
    if part is np.real:
     
        # surface E1
        ext_piece1 = [X[:nx,...].min(),
                      X[:nx,...].max(),
                      Y[:nx,...].min(),
                      Y[:nx,...].max(),
                      -1, 0]
        
        ext_piece2 = [X[nx:,...].min(),
                      X[nx:,...].max(),
                      Y[nx:,...].min(),
                      Y[nx:,...].max(),
                      0, 1]
        

        E1s1 = mlab.surf(X[:nx,...], Y[:nx,...], E1[:nx,...],
                  extent=ext_piece1,
                  opacity=0.85,
                  color=blue,
                  vmin=-1, vmax=1)
        E1s2 = mlab.surf(X[nx:,...], Y[nx:,...], E1[nx:,...],
                  extent=ext_piece2,
                  opacity=0.85,
                  color=blue,
                  vmin=-1, vmax=1)
        E1s3 = mlab.surf(X[nx-1:nx+1,:ny],
                  Y[nx-1:nx+1,:ny],
                  E1[nx-1:nx+1,:ny],
                  #extent=ext,
                  opacity=0.85,
                  color=blue,
                  vmin=-1, vmax=1)
        
        #E1s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        #E1s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        #E1s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        
        # surface E0
        ext_piece1[-2:] = 0, 1
        ext_piece2[-2:] = -1, 0
        
        E0s1 = mlab.surf(X[:nx,...], Y[:nx,...], E0[:nx,...],
                  opacity=0.85,
                  extent=ext_piece1,
                  color=red,
                  vmin=-0.1, vmax=1)
                  #vmin=-1, vmax=1)
        E0s2 = mlab.surf(X[nx:,...], Y[nx:,...], E0[nx:,...],
                  opacity=0.85,
                  extent=ext_piece2,
                  color=red,
                  vmin=-1, vmax=1)
        E0s3 = mlab.surf(X[nx-1:nx+1,:ny],
                  Y[nx-1:nx+1,:ny],
                  E0[nx-1:nx+1,:ny],
                  opacity=0.85,
                  color=red,
                  vmin=-1, vmax=1)
        
        #E0s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        #E0s1.module_manager.scalar_lut_manager.reverse_lut = True
        #
        #E0s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        #E0s2.module_manager.scalar_lut_manager.reverse_lut = True
        #
        #E0s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]*255
    
    else:   
        ext_piece1 = [X.min(),X.max(),
                      Y.min(),Y.max(),
                      -0.5, 0]
        
        ext_piece2 = [X.min(),X.max(),
                      Y.min(),Y.max(),
                      0, 0.5]
        
        #ext_piece1 = [-0.5,0.5,-0.5,0.5,-1,0]
        #ext_piece2 = [-0.5,0.5,-0.5,0.5,0,1]
        

        s1 = mlab.surf(X, Y, E1,
                        #scalars=X,
                        extent=ext_piece2,
                        representation='surface',
                        opacity=0.8, #0.90,
                        #colormap='RdBu',
                        color=red,
                        vmin=-1, vmax=1)
        delta_z = 0.03
        s2 = mlab.surf(X, Y, E0, #+delta_z,
                        #scalars=X[::-1],
                        extent=ext_piece1,
                        representation='surface',
                        opacity=0.80,
                        #colormap='RdBu',
                        color=blue,
                        vmin=-1, vmax=1)
        
        #s1.actor.property.specular = 0.45
        #s1.actor.property.specular_power = 5
        #s2.actor.property.specular = 0.45
        #s2.actor.property.specular_power = 5
        #
        #lut1 = s1.module_manager.scalar_lut_manager.lut.table.to_array()
        #lut2 = s2.module_manager.scalar_lut_manager.lut.table.to_array()
        
        #s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        #s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom*255
        
        mlab.view(130, -70, 6.5)
    
    E1 = part(OM.eVals[:,0])
    E2 = part(OM.eVals[:,1])
    z = map_trajectory(c1, c2, E1, E2)
    
    ext = ext_piece1
    ext[-2:] = -0.5, 0.5
    
    ext3d = [x.min(), x.max(), y.min(), y.max(),z.min(),z.max()]
    ext3d = [x.min(), x.max(), y.min(), y.max(),-0.5,0.5]
    mlab.plot3d(x, y, z,
                #line_width=.025,
                extent=ext3d,
                tube_radius=0.001
               )
    
    mlab.points3d(x[0], y[0], z[0],
                  color=(0, 0, 0),
                  #extent=ext,
                  scale_factor=0.01,
                  mode='sphere',
                  )
    
    u, v, w = [ np.gradient(n) for n in x, y, z ]
    x, y, z, u, v, w = [ n[-1:] for n in x, y, z, u, v, w ]
    
    mlab.quiver3d(x, y, z, u, v, w,
                  color=(0, 0, 0),
                  scale_factor=50,
                  #extent=ext,
                  mode='cone',        
                  )
        
    mlab.outline(extent=ext,
                 line_width=2.5,
                 color=(0.5, 0.5, 0.5))
    
    mlab.view(142, -72, 7.5)
    mlab.draw()
    #fig.scene.render_window.aa_frames = 8
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
                "init_state": 'b', 
                "init_loop_phase": 1*pi, 
                "loop_direction": '-',
                "calc_adiabatic_state": True
                }
        
        plot_riemann_sheets(**params)
