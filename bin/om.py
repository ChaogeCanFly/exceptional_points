#!/usr/bin/env python2.7

import brewer2mpl as brew
from ep.helpers import map_trajectory, get_height_profile
from ep.optomech import OptoMech
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from numpy import pi
    
def plot_riemann_sheets(part=np.real,
                        scale=3, #6.5
                        wireframe_skip=5.,
                        xN=153, yN=152, **kwargs):
    """Plot local Riemann sheet structure of the OM Hamiltonian."""

    #xN, yN = 31, 31
    #part = np.imag
    #part = np.real
    
    OM = OptoMech(**kwargs)
    x, y = OM.get_cycle_parameters(OM.t)
    _, c1, c2 = OM.solve_ODE()
    
    e1 = part(OM.eVals[:,0])
    e2 = part(OM.eVals[:,1])
    z = map_trajectory(c1, c2, e1, e2)
    #z = e1
    
    ############################################################################
    # two trajectories
    import copy
    kwargs2 = copy.deepcopy(kwargs)
    kwargs2['init_state'] = 'b'
    kwargs2['loop_direction'] = '+'
    OM2 = OptoMech(**kwargs2)
    xT, yT = OM2.get_cycle_parameters(OM2.t)
    _, c1T, c2T = OM2.solve_ODE()
    
    e1T = part(OM.eVals[:,0])
    e2T = part(OM.eVals[:,1])
    zT = map_trajectory(c1T, c2T, e1T, e2T)
    ############################################################################
    

    
    X, Y, Z = OM.sample_H(xN=xN, yN=yN)
    E1 = part(Z[...,1])
    E0 = part(Z[...,0])
    
    nx = np.sqrt(len(E1.ravel())).astype(int)/2
    ny = nx
    
    red = tuple(map(lambda x: x/255., (228,26,28)))
    blue = tuple(map(lambda x: x/255., (55,126,184)))
    def get_custom_cmap():
        cmap = LinearSegmentedColormap.from_list('RdBu_custom',
                                                 [red, blue], N=256)
        return cmap(np.arange(256))*255.
    
    RdBu_custom = get_custom_cmap()
    

    fig = mlab.figure(size=(1400,1000), bgcolor=(1,1,1))
    
    line_color = (0.25, 0.25, 0.25)
    arrow_ending = 750
    mlab.plot3d(x[:-arrow_ending], y[:-arrow_ending], z[:-arrow_ending]/scale,
                color=line_color,
                opacity=1.,
                tube_radius=0.003)
    ############################################################################
    # two trajectories
    line_colorT = (0.75, 0.75, 0.75)
    mlab.plot3d(xT[:-arrow_ending], yT[:-arrow_ending], zT[:-arrow_ending]/scale,
                color=(0.5,0.5,0.5),
                opacity=1.,
                tube_radius=0.003)
    ############################################################################
    
    W_Gauss_Fermi, W_Fermi, wmax, wmin = get_height_profile(X, Y, rho_y=1e-2, 
                                                            sigma_x=1e-4)
    
    
    if part is np.real:
        
        E1s1p1 = mlab.mesh(X[:nx+1,:ny+1],
                         Y[:nx+1,:ny+1],
                         E1[:nx+1,:ny+1]/scale,
                         scalars=W_Gauss_Fermi[:nx+1,:ny+1],
                         opacity=0.8,
                         #color=blue,
                         vmin=wmin, vmax=wmax)        
        E1s2 = mlab.mesh(X[:nx+1,ny:],
                         Y[:nx+1,ny:],
                         E1[:nx+1,ny:]/scale,
                         scalars=W_Fermi[:nx+1,ny:],         
                         opacity=0.8,
                         #color=blue,
                         vmin=wmin, vmax=wmax)
        # <!-- CONE
        ###E0s1 = mlab.mesh(X[:nx+1,:ny+1],
        ###                 Y[:nx+1,:ny+1],
        ###                 E0[:nx+1,:ny+1]/scale,
        ###                 scalars=W_Gauss_Fermi[:nx+1,:ny+1],
        ###                 opacity=0.8,
        ###                 color=red,
        ###                 vmin=wmin, vmax=wmax)
        ###E0s11 = mlab.mesh(X[nx+1:,:ny+1],
        ###                 Y[nx+1:,:ny+1],
        ###                 E0[nx+1:,:ny+1]/scale,
        ###                 scalars=W_Gauss_Fermi[nx+1:,:ny+1],
        ###                 opacity=0.8,
        ###                 color=blue,
        ###                 vmin=wmin, vmax=wmax)
        # --> CONE
        
        E0s1 = mlab.mesh(X[...,:ny+1],
                         Y[...,:ny+1],
                         E0[...,:ny+1]/scale,
                         scalars=W_Gauss_Fermi[...,:ny+1],
                         opacity=0.8,
                         vmin=wmin, vmax=wmax)
        E0s2 = mlab.mesh(X[:nx+1,ny:],
                         Y[:nx+1,ny:],
                         E0[:nx+1,ny:]/scale,
                         scalars=W_Fermi[:nx+1,ny:],         
                         opacity=0.8,
                         #color=red,
                         vmin=wmin, vmax=wmax)
        E0s3 = mlab.mesh(X[nx+1:,ny:],
                         Y[nx+1:,ny:],
                         E0[nx+1:,ny:]/scale,
                         scalars=W_Fermi[nx+1:,ny:],
                         opacity=0.8,
                         #color=blue,
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


        E1s1p2 = mlab.mesh(X[nx+1:,:ny+1],
                         Y[nx+1:,:ny+1],
                         E1[nx+1:,:ny+1]/scale,
                         scalars=W_Gauss_Fermi[nx+1:,:ny+1],
                         opacity=0.8,
                         #color=red, 
                         vmin=wmin, vmax=wmax)
        E1s3 = mlab.mesh(X[nx+1:,ny:],
                         Y[nx+1:,ny:],
                         E1[nx+1:,ny:]/scale,
                         scalars=W_Fermi[nx+1:,ny:],
                         opacity=0.8,
                         #color=red, 
                         vmin=wmin, vmax=wmax)

        
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
        
    else:
        s2 = mlab.mesh(X, Y, E0/scale,
                        scalars=W_Fermi, 
                        representation='surface',
                        opacity=0.8,
                        #color=blue,
                        vmin=-wmin, vmax=wmax)        
        s1 = mlab.mesh(X, Y, E1/scale,
                        scalars=W_Fermi, 
                        representation='surface',
                        opacity=0.8,
                        #color=red,
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
                  scale_factor=0.0125,
                  mode='sphere')
    
    u, v, w = [ np.gradient(n) for n in x, y, z/scale ]
    x, y, z, u, v, w = [ n[-arrow_ending] for n in x, y, z/scale, u, v, w ]
    mlab.quiver3d(x, y, z, u, v, w,
                color=line_color,
                scale_factor=0.025,
                resolution=200,
                mode='cone',
                scale_mode='scalar'
                )
    
    ############################################################################
    # two trajectories
    mlab.points3d(xT[0], yT[0], zT[0]/scale,
                  color=line_colorT,
                  scale_factor=0.0125,
                  mode='sphere')
    
    uT, vT, wT = [ np.gradient(n) for n in xT, yT, zT/scale ]
    xT, yT, zT, uT, vT, wT = [ n[-arrow_ending] for n in xT, yT, zT/scale, uT, vT, wT ]
    mlab.quiver3d(xT, yT, zT, uT, vT, wT,
                color=line_colorT,
                scale_factor=0.025,
                resolution=200,
                mode='cone',
                scale_mode='scalar'
                )
    ############################################################################
    
    mlab.axes(figure=fig,
              color=(0.25,0.25,0.25),
              extent=[X.min(),X.max(),
                      Y.min()*0.95,Y.max()*0.95,
                      E0.min()/3.5,E1.max()/3.5])
    
    engine = mlab.get_engine()
    
    if part is np.imag:
        scene = engine.scenes[0]
        scene.scene.camera.position = [0.34763136753239232, -0.37268098883113926, 0.19025556632156901]
        scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.10361763966271739, 0.17406840710399826, 0.97926685556032378]
        scene.scene.camera.clipping_range = [0.57179585985024317, 1.4283725482790282]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()
    elif part is np.real:
        scene = engine.scenes[0]
        scene.scene.camera.position = [0.34763136753239232, -0.37268098883113926, 0.19025556632156901]
        scene.scene.camera.position = [0.34763136753239232, -0.37268098883113926, 0.5025556632156901]
        scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.10361763966271739, 0.17406840710399826, 0.97926685556032378]
        scene.scene.camera.clipping_range = [0.57179585985024317, 1.4283725482790282]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

    if part is np.imag:
        pass
        #surface1 = engine.scenes[0].children[1].children[0].children[0].children[0]
        #surface1.actor.property.backface_culling = True
        #surface2 = engine.scenes[0].children[2].children[0].children[0].children[0]
        #surface2.actor.property.frontface_culling = True
    elif part is np.real:
        # frontface culling of map_trajectory
        pass
        #surface = engine.scenes[0].children[0].children[0].children[0].children[0].children[0]
        #surface.actor.property.frontface_culling = True
        #surface1 = engine.scenes[0].children[1].children[0].children[0].children[0]
        #surface1.actor.property.backface_culling = True
        #surface2 = engine.scenes[0].children[2].children[0].children[0].children[0]
        #surface2.actor.property.backface_culling = True
        #surface3 = engine.scenes[0].children[3].children[0].children[0].children[0]
        #surface3.actor.property.backface_culling = True
        #surface5 = engine.scenes[0].children[5].children[0].children[0].children[0]
        #surface5.actor.property.backface_culling = True
        #surface6 = engine.scenes[0].children[6].children[0].children[0].children[0].children[0]
        #surface6.actor.property.backface_culling = True
        #surface11 = engine.scenes[0].children[11].children[0].children[0].children[0].children[0]
        #surface11.actor.property.backface_culling = True


    # settings for Fig. 2a) and 2b)
    if 1:
        scene = engine.scenes[0]
        scene.scene.camera.position = [0.39731242769716185, 1.2410028647957307, 0.42627834802260051]
        scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.16459501739522236, -0.41992228721278874, 0.89250980552072745]
        scene.scene.camera.clipping_range = [0.53243401166881033, 1.4779210054119616]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()
    
    # settings for Fig. 2b)
    
    fig.scene.render_window.aa_frames = 8
    #mlab.show()
    if part is np.real:
        str_part = "real"
    else:
        str_part = "imag"
    mlab.savefig("{}.png".format(str_part))
    mlab.axes(x_axis_visibility=False,
              y_axis_visibility=False,
              z_axis_visibility=False)
    mlab.savefig("{}_no_axes.png".format(str_part))



def plot_figures(fig='2a', part='imag', direction='-',
                 T=45., R=0.1, gamma=1.):
    
    import subprocess
    
    params = {
                "T": T, 
                "R": R, 
                "gamma": gamma
    }
    
    if fig == '2a':
        settings = {
                "init_state": 'b', 
                "init_phase": 0, 
                "loop_direction": '-',
                }
    elif fig == '2b':
        settings = {
                "init_state": 'a', 
                "init_phase": 0, 
                "loop_direction": '+',
                }            
    elif fig == '2c':
        settings = {
                "init_state": 'b', 
                "init_phase": pi, 
                "loop_direction": '-',
                }
    
    params.update(settings)
    
    if part == 'imag':
        params['part'] = np.imag
    else:
        params['part'] = np.real
        
    plot_riemann_sheets(**params)
    
    for f in part, part + '_no_axes':
        infile = f + '.png'
        outfile = 'Fig{}_{}{}_new.png'.format(fig,f,direction)
        call = ['convert', '-transparent', 'white', '-trim', infile, outfile]
        subprocess.check_call(call)
    
    
if __name__ == '__main__':
    print "Warning: is normalization symmetric?"
    
    import argh
    argh.dispatch_command(plot_figures)
    
