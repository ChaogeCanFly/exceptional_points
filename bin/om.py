#!/usr/bin/env python2.7

from matplotlib.colors import LinearSegmentedColormap
import mayavi.mlab as mlab
import numpy as np
from numpy import pi

from ep.helpers import map_trajectory, get_height_profile
from ep.optomech import OptoMech


def get_custom_cmap(clist):
    cmap = LinearSegmentedColormap.from_list('RdBu_custom', clist, N=256)
    return cmap(np.arange(256))*255.


def plot_riemann_sheets(part=np.real,
                        scale=4.0,  # 3 <- paper plots, #6.5
                        show=False,
                        colorgradient=False,
                        fignum='1c',
                        adiabatic=False,
                        trajectory=True,
                        parallel_projection=False,
                        opacity=1.0,
                        xN=153, yN=152, **kwargs):
    """Plot local Riemann sheet structure of the OM Hamiltonian."""

    wireframe_skip = xN/25.

    if part is np.real:
        scale = 4.0
        scale = 3.25
    else:
        scale = 3.0

    if fignum == 'DP':
        scale = 1.0

    OM = OptoMech(**kwargs)
    x, y = OM.get_cycle_parameters(OM.t)
    _, c1, c2 = OM.solve_ODE()

    e1 = part(OM.eVals[:, 0])
    e2 = part(OM.eVals[:, 1])
    if adiabatic:
        z = e1
    else:
        z = map_trajectory(c1, c2, e1, e2)

    if fignum == 'DP':
        dx = 0.05
        dy = dx
        X, Y, Z = OM.sample_H(xN=xN, yN=yN,
                              xmin=-dx, xmax=dx,
                              ymin=-dy, ymax=dy)
    else:
        dx = 0.11
        dy = dx
        X, Y, Z = OM.sample_H(xN=xN, yN=yN,
                              xmin=-dx, xmax=dx,
                              ymin=0.5-dy, ymax=0.5+dy)

    E1 = part(Z[..., 1])
    E0 = part(Z[..., 0])

    nx = np.sqrt(len(E1.ravel())).astype(int)/2
    ny = nx

    # red blue
    red = tuple(map(lambda x: x/255., (228, 26, 28)))
    blue = tuple(map(lambda x: x/255., (55, 126, 184)))
    # green orange (Dark1)
    # red = tuple(map(lambda x: x/255., (217, 95, 2)))
    # blue = tuple(map(lambda x: x/255., (27, 158, 119)))
    # green purple (Set1)
    # red = tuple(map(lambda x: x/255., (77, 175, 74)))
    # blue = tuple(map(lambda x: x/255., (152, 78, 163)))
    # green orange
    # red = tuple(map(lambda x: x/255., (255, 117, 24)))
    # blue = tuple(map(lambda x: x/255., (0, 128, 128)))
    # green orange
    # red = tuple(map(lambda x: x/255., (204, 85, 0)))
    # blue = tuple(map(lambda x: x/255., (0, 128, 128)))
    # blue yellow
    # red = tuple(map(lambda x: x/255., (218, 165, 32)))
    # blue = tuple(map(lambda x: x/255., (0, 135, 189)))

    RdBu_custom = get_custom_cmap(clist=[red, blue])
    if colorgradient:
        red, blue = None, None

    fig = mlab.figure(size=(1400, 1000), bgcolor=(1, 1, 1))

    if trajectory:
        if fignum == 'DP':
            tube_radius = 0.004
        else:
            #tube_radius = 0.05
            tube_radius = 0.005

        arrow_end = OM.tN/10
        arrow_end = OM.tN/15
        line_color = (0.25, 0.25, 0.25)
        mlab.plot3d(x[:-arrow_end], y[:-arrow_end], z[:-arrow_end]/scale,
                    color=line_color,
                    opacity=1.,
                    tube_radius=tube_radius)

    W_Gauss_Fermi, W_Fermi, wmax, wmin = get_height_profile(X, Y,
                                                            rho_y=1e-2,
                                                            sigma_x=1e-4)
    if part is np.real:
        E1s1p1 = mlab.mesh(X[:nx+1, :ny+1],
                           Y[:nx+1, :ny+1],
                           E1[:nx+1, :ny+1]/scale,
                           scalars=W_Gauss_Fermi[:nx+1, :ny+1],
                           opacity=opacity,
                           # color=blue,
                           vmin=wmin, vmax=wmax)
        E1s2 = mlab.mesh(X[:nx+1, ny:],
                         Y[:nx+1, ny:],
                         E1[:nx+1, ny:]/scale,
                         scalars=W_Fermi[:nx+1, ny:],
                         opacity=opacity,
                         # color=blue,
                         vmin=wmin, vmax=wmax)
        # <!-- CONE
        if fignum == 'DP':
            E0s1 = mlab.mesh(X[:nx+1, :ny+1],
                             Y[:nx+1, :ny+1],
                             E0[:nx+1, :ny+1]/scale,
                             scalars=W_Gauss_Fermi[:nx+1, :ny+1],
                             opacity=opacity,
                             # color=red,
                             vmin=wmin, vmax=wmax)
            E0s11 = mlab.mesh(X[nx+1:, :ny+1],
                              Y[nx+1:, :ny+1],
                              E0[nx+1:, :ny+1]/scale,
                              scalars=W_Gauss_Fermi[nx+1:, :ny+1],
                              opacity=opacity,
                              # color=blue,
                              vmin=wmin, vmax=wmax)
        # --> CONE
        else:
            E0s1 = mlab.mesh(X[..., :ny+1],
                             Y[..., :ny+1],
                             E0[..., :ny+1]/scale,
                             scalars=W_Gauss_Fermi[..., :ny+1],
                             opacity=opacity,
                             vmin=wmin,
                             vmax=wmax)
        E0s2 = mlab.mesh(X[:nx+1, ny:],
                         Y[:nx+1, ny:],
                         E0[:nx+1, ny:]/scale,
                         scalars=W_Fermi[:nx+1, ny:],
                         opacity=opacity,
                         vmin=wmin, vmax=wmax)
        E0s3 = mlab.mesh(X[nx+1:, ny:],
                         Y[nx+1:, ny:],
                         E0[nx+1:, ny:]/scale,
                         scalars=W_Fermi[nx+1:, ny:],
                         opacity=opacity,
                         vmin=wmin, vmax=wmax)

        if fignum == 'DP':
            Red = get_custom_cmap(clist=[red, red])
            Blue = get_custom_cmap(clist=[blue, blue])
            E0s1.module_manager.scalar_lut_manager.lut.table = Red
            E0s11.module_manager.scalar_lut_manager.lut.table = Blue
            E0s2.module_manager.scalar_lut_manager.lut.table = Red
            E0s3.module_manager.scalar_lut_manager.lut.table = Blue
        else:
            E0s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
            E0s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
            E0s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom

        mlab.surf(X[nx+1::wireframe_skip, ::wireframe_skip],
                  Y[nx+1::wireframe_skip, ::wireframe_skip],
                  E0[nx+1::wireframe_skip, ::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0, 0, 0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)

        mlab.surf(X[:nx+2:wireframe_skip, ::wireframe_skip],
                  Y[:nx+2:wireframe_skip, ::wireframe_skip],
                  E0[:nx+2:wireframe_skip, ::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0, 0, 0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)

        E1s1p2 = mlab.mesh(X[nx+1:, :ny+1],
                           Y[nx+1:, :ny+1],
                           E1[nx+1:, :ny+1]/scale,
                           scalars=W_Gauss_Fermi[nx+1:, :ny+1],
                           opacity=opacity,
                           vmin=wmin, vmax=wmax)
        E1s3 = mlab.mesh(X[nx+1:, ny:],
                         Y[nx+1:, ny:],
                         E1[nx+1:, ny:]/scale,
                         scalars=W_Fermi[nx+1:, ny:],
                         opacity=opacity,
                         vmin=wmin, vmax=wmax)

        if fignum == 'DP':
            E1s1p1.module_manager.scalar_lut_manager.lut.table = Blue
            E1s1p2.module_manager.scalar_lut_manager.lut.table = Red
            E1s2.module_manager.scalar_lut_manager.lut.table = Blue
            E1s3.module_manager.scalar_lut_manager.lut.table = Red
        else:
            E1s1p1.module_manager.scalar_lut_manager.lut.table = RdBu_custom
            E1s1p2.module_manager.scalar_lut_manager.lut.table = RdBu_custom
            E1s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom
            E1s3.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]

        mlab.surf(X[nx+1::wireframe_skip, ::wireframe_skip],
                  Y[nx+1::wireframe_skip, ::wireframe_skip],
                  E1[nx+1::wireframe_skip, ::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0, 0, 0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)

        mlab.surf(X[:nx+2:wireframe_skip, ::wireframe_skip],
                  Y[:nx+2:wireframe_skip, ::wireframe_skip],
                  E1[:nx+2:wireframe_skip, ::wireframe_skip]/scale,
                  opacity=0.05,
                  color=(0, 0, 0),
                  representation='wireframe',
                  vmin=wmin, vmax=wmax)

    else:
        s2 = mlab.mesh(X, Y, E0/scale,
                       scalars=W_Fermi,
                       representation='surface',
                       opacity=opacity,
                       color=blue,
                       vmin=-wmin, vmax=wmax)
        s1 = mlab.mesh(X, Y, E1/scale,
                       scalars=W_Fermi,
                       representation='surface',
                       opacity=opacity,
                       color=red,
                       vmin=-wmin, vmax=wmax)

        s1.module_manager.scalar_lut_manager.lut.table = RdBu_custom[::-1]
        s2.module_manager.scalar_lut_manager.lut.table = RdBu_custom

        mlab.surf(X[::wireframe_skip, ::wireframe_skip],
                  Y[::wireframe_skip, ::wireframe_skip],
                  E1[::wireframe_skip, ::wireframe_skip]/scale,
                  representation='wireframe',
                  opacity=0.05,
                  color=(0, 0, 0),
                  vmin=-wmin, vmax=wmax)

        mlab.surf(X[::wireframe_skip, ::wireframe_skip],
                  Y[::wireframe_skip, ::wireframe_skip],
                  E0[::wireframe_skip, ::wireframe_skip]/scale,
                  representation='wireframe',
                  opacity=0.05,
                  color=(0, 0, 0),
                  vmin=-wmin, vmax=wmax)

    for E in E0, E1:
        for xx, yy, zz in ([X[:, 0],  Y[:, 0],  E[:, 0]/scale],
                           [X[:, -1], Y[:, -1], E[:, -1]/scale],
                           [X[0, :],  Y[0, :],  E[0, :]/scale],
                           [X[-1, :], Y[-1, :], E[-1, :]/scale]):
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

    if trajectory:
        if fignum == 'DP':
            scale_factor_point = 0.015
            scale_factor_quiver = 0.0175
        else:
            scale_factor_point = 0.022
            scale_factor_quiver = 0.05

        mlab.points3d(x[0], y[0], z[0]/scale,
                      color=line_color,
                      scale_factor=scale_factor_point,
                      mode='sphere')

        u, v, w = [np.gradient(m) for m in x, y, z/scale]
        x, y, z, u, v, w = [m[-arrow_end] for m in x, y, z/scale, u, v, w]
        mlab.quiver3d(x, y, z, u, v, w,
                      color=line_color,
                      scale_factor=scale_factor_quiver,
                      resolution=500,
                      mode='cone',
                      scale_mode='scalar')

    ###########################################################################
    # quiver settings
    engine = mlab.get_engine()
    if trajectory and not fignum == 'DP':
        try:
            vectors = engine.scenes[0].children[14].children[0].children[0]
            vectors.glyph.glyph_source.glyph_source.direction = np.array([0., 0., 0.])
        except:
            vectors = engine.scenes[0].children[23].children[0].children[0]
            vectors.glyph.glyph_source.glyph_source.direction = np.array([0., 0., 0.])

        vectors.glyph.glyph_source.glyph_position = 'center'
        vectors.glyph.glyph_source.glyph_source.angle = 18.0
        vectors.glyph.glyph_source.glyph_source.center = np.array([0.2, 0., 0.])
        vectors.glyph.glyph_source.glyph_source.height = 0.81
        vectors.glyph.glyph_source.glyph_source.radius = 0.27
    elif trajectory and fignum == 'DP':
        vectors = engine.scenes[0].children[26].children[0].children[0]
        vectors.glyph.glyph_source.glyph_source.direction = np.array([ 1.,  0.,  0.])
        vectors.glyph.glyph_source.glyph_source.center = np.array([ 0.5,  0. ,  0. ])
        vectors.glyph.glyph_source.glyph_source.radius = 0.6
        vectors.glyph.glyph_source.glyph_source.progress = 1.0
        vectors.glyph.glyph_source.glyph_source.angle = 20.
    ###########################################################################

    mlab.axes(figure=fig,
              color=(0.25, 0.25, 0.25),
              extent=[X.min(), X.max(),
                      Y.min()*0.95, Y.max()*0.95,
                      E0.min()/3.5, E1.max()/3.5])
    engine = mlab.get_engine()

    print "figure", fignum
    if fignum == '1c' or fignum == '1c_alt' or fignum == 'DP':
        if part is np.real:
            # scale=4
            scene = engine.scenes[0]
            scene.scene.camera.position = [0.62549444644128371, 0.91502339993579351, 0.57551747665335107]
            scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [-0.38022273275020868, -0.49725772693819542, 0.7798496178752814]
            scene.scene.camera.clipping_range = [0.52540000719025637, 1.4867753678334952]
            scene.scene.camera.compute_view_plane_normal()
            scene.scene.render()
            #
            scene.scene.camera.position = [0.56093203728033203, 0.89320687991303216, 0.65257701209600438]
            scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [-0.44478871574781131, -0.53938487591390027, 0.71500136641740708]
            scene.scene.camera.clipping_range = [0.51554383800662118, 1.4991822541676811]
            scene.scene.camera.compute_view_plane_normal()
            scene.scene.render()
        else:
            # scale = 3
            scene = engine.scenes[0]
            scene.scene.camera.position = [0.60185186146516323, 0.87460659588386158, 0.62681954629275272]
            scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [-0.48467401632134571, -0.45745893240421753, 0.7455349911751491]
            scene.scene.camera.clipping_range = [0.53148188620606163, 1.4791195352031188]
            scene.scene.camera.compute_view_plane_normal()
            scene.scene.render()
    elif "wg" in fignum:
        scene = engine.scenes[0]
        camera_light3 = engine.scenes[0].scene.light_manager.lights[3]
        camera_light3.activate = True
        camera_light3.intensity = 0.5
        camera_light3.azimuth = -100.0
        camera_light3.elevation = 30.0
        scene.scene.camera.position = [-0.58666637486265438, 0.34190933964282821, 0.46278176181631675]
        scene.scene.camera.focal_point = [0.0, 0.5, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [0.54725173023115548, 0.27905704792369884, 0.78907712409061603]
        scene.scene.camera.clipping_range = [0.38983416938417631, 1.231847141331484]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.parallel_projection = parallel_projection
        scene.scene.render()
    elif fignum == '2a' or fignum == '2b':
        # scale=3.5
        scene = engine.scenes[0]
        scene.scene.camera.position = [0.39731242769716185, 1.2410028647957307, 0.42627834802260051]
        scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.16459501739522236, -0.41992228721278874, 0.89250980552072745]
        scene.scene.camera.clipping_range = [0.53243401166881033, 1.4779210054119616]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        # 2014-09-4
        scene = engine.scenes[0]
        scene.scene.camera.position = [0.35754482670282456, 1.3417854385384826, 0.22437331932214213]
        scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.052643847291822997, -0.23383567551320561, 0.97084988654250659]
        scene.scene.camera.clipping_range = [0.55919231561031824, 1.4442378137670175]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        if part is np.imag:
            scene = engine.scenes[0]
            scene.scene.camera.position = [0.33009998678631336, 1.2557918887354751, 0.45628244887314351]
            scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [-0.14888057078322309, -0.45833306997754164, 0.87622221645437848]
            scene.scene.camera.clipping_range = [0.53805389345878574, 1.4708467321035021]
            scene.scene.camera.compute_view_plane_normal()
            scene.scene.render()
    elif fignum == '2c':
        # 2014-09-4
        scene = engine.scenes[0]
        scene.scene.camera.position = [0.35754482670282456, 1.3417854385384826, 0.22437331932214213]
        scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.052643847291822997, -0.23383567551320561, 0.97084988654250659]
        scene.scene.camera.clipping_range = [0.55919231561031824, 1.4442378137670175]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        # alternative: 2014-10-2
        scene = engine.scenes[0]
        scene.scene.camera.position = [-0.35265479615669348, -0.059068820705388034, 0.5321422968823194]
        scene.scene.camera.focal_point = [0.0, 0.4907165063509461, 0.0]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [0.19456992935879738, 0.61489761529696563, 0.76422736491924792]
        scene.scene.camera.clipping_range = [0.40523852173561664, 1.3949522401256085]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        if part is np.imag:
            # 2014-09-4
            scene = engine.scenes[0]
            scene.scene.camera.position = [-0.28059939494943476, -0.39585508722999119, 0.19430587084751583]
            scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [0.065714500791410779, 0.19373176259766278, 0.97885116771986258]
            scene.scene.camera.clipping_range = [0.58142164617061654, 1.4162556665039547]
            scene.scene.camera.compute_view_plane_normal()
            scene.scene.render()

            scene = engine.scenes[0]
            scene.scene.camera.position = [-0.28965117077674074, -0.40691928995030441, 0.11336736163262609]
            scene.scene.camera.focal_point = [0.0, 0.49071651625633239, 0.0]
            scene.scene.camera.view_angle = 30.0
            scene.scene.camera.view_up = [0.037959376611672771, 0.11314407332158574, 0.99285321392411918]
            scene.scene.camera.clipping_range = [0.59587171711805942, 1.3980660043314275]
            scene.scene.camera.compute_view_plane_normal()
            scene.scene.render()

    # 2014-09-23: trajectory default normal fix
    if trajectory:
        tube = engine.scenes[0].children[0].children[0].children[0]
        tube.filter.default_normal = np.array([0.,  0.,  1.])
        tube.filter.use_default_normal = True
    ####

    if show:
        mlab.show()
    else:
        fig.scene.render_window.aa_frames = 20
        if part is np.real:
            str_part = "real"
        else:
            str_part = "imag"
        mlab.axes(x_axis_visibility=False,
                  y_axis_visibility=False,
                  z_axis_visibility=False)
        mlab.savefig("{}_no_axes.png".format(str_part))


def plot_figures(fignum='2a', part='imag', direction='-', show=False,
                 colorgradient=False, T=45., R=0.1, gamma=1., adiabatic=False,
                 opacity=1.0, trajectory=True, init_state='b',
                 parallel_projection=False, xN=351, yN=352):

    import subprocess

    params = {"T": T,
              "R": R,
              "gamma": gamma,
              "fignum": fignum,
              "show": show,
              "colorgradient": colorgradient,
              "trajectory": trajectory,
              "parallel_projection": parallel_projection,
              "opacity": opacity,
              "adiabatic": adiabatic,
              "xN": xN,
              "yN": yN}

    if fignum == '1c':
        params['R'] = 0.9*R
        settings = {"init_state": 'b',
                    "init_phase": pi/2,
                    "loop_direction": '-'}
    elif fignum == '1c_alt':
        params['R'] = 0.9*R
        settings = {"init_state": init_state,
                    "init_phase": pi/2,
                    "loop_direction": direction, }
    elif fignum == 'wg':
        params['R'] = 0.9*R
        settings = {"init_state": init_state,
                    "init_phase": 0.01,
                    "loop_direction": direction, }
    elif fignum == 'wg_alt':
        params['R'] = 0.9*R
        settings = {"init_state": init_state,
                    "init_phase": -0.75,
                    "loop_direction": direction, }
    elif fignum == 'wg_alt_2':
        params['R'] = 0.9*R
        settings = {"init_state": init_state,
                    "init_phase": -1.95,
                    "loop_direction": direction, }
    elif fignum == '2a':
        settings = {"init_state": 'a',
                    "init_phase": 0,
                    "loop_direction": '+'}
    elif fignum == '2b':
        settings = {"init_state": 'b',
                    "init_phase": 0,
                    "loop_direction": '-'}
    elif fignum == '2c':
        settings = {"init_state": 'b',
                    "init_phase": pi,
                    "loop_direction": '-'}
    elif fignum == 'DP':
        settings = {"init_state": 'b',
                    "init_phase": pi,
                    "loop_direction": '-',
                    "loop_type": "RAP",
                    "x_R0": 1.1*R,
                    "y_R0": 0.048}

    params.update(settings)

    if part == 'imag':
        params['part'] = np.imag
    else:
        params['part'] = np.real

    plot_riemann_sheets(**params)

    for f in part, part + '_no_axes':
        infile = f + '.png'
        outfile = 'Fig{}_{}_{}_{}.png'.format(fignum, f, init_state, direction)
        try:
            cmd = 'convert -transparent white -trim {} {}'.format(infile, outfile)
            subprocess.check_call(cmd.split())
        except:
            pass


if __name__ == '__main__':
    import argh
    argh.dispatch_command(plot_figures)
