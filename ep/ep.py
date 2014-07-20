#!/usr/bin/env python

from ep.toymodel import Toymodel


def circle_EP():
    """
    Plot the projected amplitudes, the corresponding adiabatic
    predictions vs. time. Additionally, the real and imaginary
    part of the energy spectrum is plotted, as well as the distance
    of the trajectory in parameter space from the EP.
    """
    for init_loop_phase in 5, : #range(13):
    #for init_loop_phase in (0, ):
    
        p0 = init_loop_phase*2.*pi/12.
        
        params = getParamList(a2 = 0.0, b1 = 0.8, b2 = 0.8,
                              c1 = 0.1, c2 = 0.1, p = p0)
        #params = getParamList(a2 = -0.0, b1 = 0.25, b2 = 1.,
        #                      c1 = 0.1, c2 = 0.1, p = p0)
        geometry = ["Circle", "Circle_wedge", "Circle_whirl",
                    "Dropshape", "Dropshape_whirl"]
        #geometry = ["Circle_whirl"]
        geometry = ["Circle"]
        
        T = 10
        
        for g in geometry:
            for direction in '+', '-': #, '+':
                for init in 'c', : #'b':
                    h = Toymodel(T=T, loop_type=g, init_state=init,
                               init_cond=params, loop_direction=direction)
                    
                    ##
                    # plot amplitudes a(t), b(t)
                    ##
                    subplot2grid((3,3), (0,0), colspan=2)
                    xlim(0, T)
                    ylim(1e-3, 1e5)
                    xlabel("time [a.u.]")
                    ylabel(r"Amplitudes $|c_n(t)|$")
                    t, psi_a, psi_b = h.solve_ODE()

                    a_ad, b_ad = (h.Psi_adiabatic[:,0],
                                  h.Psi_adiabatic[:,1])
                    # intial population of states a and b
                    a0 = abs(psi_a[0])
                    b0 = abs(psi_b[0])

                    semilogy(t,abs(psi_a),"r-",
                             label=r"$\langle \phi_a|\psi \rangle_t$")
                    semilogy(t,abs(psi_b),"g-",
                             label=r"$\langle \phi_b|\psi \rangle_t$")
                    semilogy(t,a0*abs(a_ad), "b--",
                             label=r"$a_{\mathrm{ad}}(t)$")
                    semilogy(t,b0*abs(b_ad), "k--",
                             label=r"$b_{\mathrm{ad}}(t)$")
                    legend(bbox_to_anchor=(1.05, 1),
                           loc=2, ncol=2, borderaxespad=0.)
                    
                    ##
                    # plot path around EP
                    ##
                    f = subplot2grid((3,3), (2,2), aspect=1)
                    f.yaxis.tick_right()
                    f.yaxis.set_label_position("right")
                    xlim(-2.1,2.1)
                    ylim(-2.1,2.1)
                    xlabel(r"$\lambda_1$")
                    ylabel(r"$\lambda_2$")
                    plot([0,0],[0,2],"ko-")
                    x0, y0 = h.get_cycle_parameters(0.)
                    x, y = h.get_cycle_parameters(h.t)
                    
                    draw_arrow(h.loop_direction, h.init_loop_phase)
                    
                    plot(x[:500], y[:500], "b-")
                    offset=100
                    quiver(x[offset:-1:offset], y[offset:-1:offset],
                           x[1+offset::offset]-x[offset:-1:offset],
                           y[1+offset::offset]-y[offset:-1:offset],
                           #angles='xy', color='r', scale_units='x', scale=2) #, scale=1)
                           #units='dots', angles='xy', scale=1e-4)
                           #scale_units='xy',
                           angles='xy', scale=None, width=5e-2, color='k')
                    plot(x0, y0, "ro")                    
                    ##
                    # plot real/imag(E1(t)), imag(E2(t))
                    ##
                    subplot2grid((3,3), (1,0), colspan=2)
                    Ea, Eb = h.eVals[:,0], h.eVals[:,1]
                    # measure for NA effects
                    # caveat: as of yet, only valid if initial state is a!
                    dGamma = imag(Ea-Eb)
                    #amax, bmax = [ abs(x).max() for x in (psi_a, psi_b) ]
                    xlim(0, T)
                    ylim(-1.6, 1.6)
                    xlabel("time [a.u.]")
                    ylabel("Energy")
                    plot(t, imag(Ea), "r-",label=r"$Im(E_a(t))$")
                    plot(t, imag(Eb), "g-",label=r"$Im(E_b(t))$") 
                    plot(t, real(Ea), "r--",label=r"$Re(E_a(t))$")
                    plot(t, real(Eb), "g--",label=r"$Re(E_b(t))$")
                    
                    t_imag = map_trajectory(abs(psi_a), abs(psi_b),
                                            imag(Ea), imag(Eb))
                    t_real = map_trajectory(abs(psi_a), abs(psi_b),
                                            real(Ea), real(Eb))
                    np.savetxt("psi_imag_%s_%s.dat" % (g, init_loop_phase), (t,t_imag))
                    np.savetxt("psi_real_%s_%s.dat" % (g, init_loop_phase), (t,t_real))
                    plot(t, t_imag, "k-")
                    plot(t, t_real, "k--")
                    
                    legend(bbox_to_anchor=(1.05, 1),
                           loc=2, borderaxespad=0.)
                                    
                    ##
                    # plot dependence on |R - R_EP|
                    ##
                    subplot2grid((3,3), (2,0), colspan=2)
                    ylim(0, 1.5)
                    xlabel("time [a.u.]")
                    ylabel(r"$|R(t) - R_\mathrm{EP}|$")
                    plot(t, np.sqrt(x**2 + y**2), "r-")

                    ##
                    # save figure
                    ##
                    savefig("%s_%i_%s_%s.png" % (g, init_loop_phase,
                                                 h.init_state, direction) )
                    clf()
                    #exit()


def plot_flip_error():
        """
        Plot the flip-error measures R1 and R2 as a function of the
        cycle time T (loop duration).
        """
        ##
        # plot ratios b_1/a_1(T)
        ##
        params = getParamList(a1 = 0.0, a2 = 0.0, b1 = 0.8, b2 = 0.8,
                          c1 = 0.2, c2 = 0.2, p = 0.0)
        params = getParamList(a2 = -0.3, b1 = 0.45, b2 = 0.9,
                              c1 = 0.1, c2 = 0.1, p = 0.0)
        Tmax = 20
        TN = 50
        Trange = np.linspace(0.01, Tmax, TN)
        R1 = np.zeros((TN,), complex)
        R2 = np.zeros((TN,), complex)
        
        for n, T in enumerate(Trange):
            print "T = ", T
            
            ##
            # flip-error R1
            ##
            h = EP_Base(T=T, loop_type=1, init_state='a',
                       init_cond=params, loop_direction='-')
            _, psi_a, psi_b = h.solve_ODE()
            R1[n] = psi_b[-1]/psi_a[-1]
            
            ##
            # flip-error R2
            ##
            h = EP_Base(T=T, loop_type=1, init_state='b',
                       init_cond=params, loop_direction='-')
            _, psi_a, psi_b = h.solve_ODE()
            R2[n] = psi_a[-1]/psi_b[-1]
            
        xlim(0, Tmax)
        ylim(1e-2, 1e3)
        semilogy(Trange, abs(R1), "ro-")
        semilogy(Trange, abs(R2), "go-")
        semilogy(Trange, abs(R1*R2), "ko-")
        show()
        
        
if __name__ == '__main__':
    circle_EP()
