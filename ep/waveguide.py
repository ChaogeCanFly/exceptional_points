#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

import brewer2mpl as brew

from ep.base import Base
from ep.dissipation import Loss


class Waveguide(Base):
    """Waveguide class."""

    def __init__(self, L=100, d=1.0, eta=0.05, N=1.5, theta=0.0,
                 x_R0=None, y_R0=None, **base_kwargs):
        """Exceptional Point (EP) waveguide class.

        Copies methods and variables from the Base class and adds new
        parameters.

            Additional parameters:
            ----------------------
                d: float
                    Waveguide width
                eta: float
                    Dissipation coefficient
                N: float
                    Number of open modes
                theta: float
                    Phase difference between upper and lower boundary
        """
        Base.__init__(self, T=L, **base_kwargs)

        self.d = d                      # wire width
        self.L = L                      # wire length
        self.eta = eta                  # dissipation coefficient

        self.theta_boundary = theta     # phase angle between upper
                                        # and lower boundary
        self.N = N                      # number of open modes
        self.k = lambda n: np.sqrt(N**2 - n**2)*np.pi
                                        # get wavevector in x-direction
        kF = N*np.pi/d                  # Fermi wavevector
        self.kF = kF

        # change trajectory positions
        if None in (x_R0, y_R0):
            self.x_R0 = self.x_EP
            self.y_R0 = self.x_EP

    def H(self, t, x=None, y=None):
        """Inherited from child classes."""
        pass

    def get_cycle_parameters(self, t):
        """Return the trajectory coordinates (x(t), y(t)) at time t."""

        x_R0, y_R0 = self.x_R0, self.y_R0
        w, phi0 = self.w, self.init_phase
        loop_type = self.loop_type

        if loop_type == "Constant":
            return x_R0, y_R0

        elif loop_type == "Constant_delta":
            return x_R0 * (1.0 - np.cos(w*t)), y_R0

        elif loop_type == "Circle":
            lambda1 = lambda t: x_R0 + x_R0*np.cos(w*t + phi0)
            lambda2 = lambda t: y_R0 + y_R0*np.sin(w*t + phi0)
            return lambda1(t), lambda2(t)

        elif loop_type == "Ellipse":
            lambda1 = lambda t: x_R0 * (1. - np.cos(w*t))
            lambda2 = lambda t: y_R0 - 8.*x_R0*np.sin(w*t) + phi0
            return lambda1(t), lambda2(t)

        elif loop_type == "Varcircle":
            lambda1 = lambda t: x_R0 * (1. - np.cos(w*t))
            lambda2 = lambda t: y_R0 - x_R0*np.sin(w*t) + phi0
            return lambda1(t), lambda2(t)

        elif loop_type == "Bell":
            sign = -int(self.loop_direction + "1")
            lambda1 = lambda t: x_R0 * (1. - np.cos(w*t))
            # take also sign change in w=2pi/T into account
            lambda2 = lambda t: y_R0 * sign * (sign*w*t/pi - 1) + phi0
            return lambda1(t), lambda2(t)

        else:
            raise Exception(("Error: loop_type {}"
                             "does not exist!").format(loop_type))

    def get_boundary(self, x=None, eps=None, delta=None, L=None,
                     d=None, kr=None, theta_boundary=None, smearing=False):
        """Get the boundary function xi as a function of the spatial coordinate x.

            Parameters:
            -----------
                x: ndarray
                    Spatial/temporal coordinate.
                eps: float
                    Boundary roughness strength.
                delta: float
                    Boundary frequency detuning.
                d: float
                    Waveguide width.
                kr: float
                    Boundary modulation frequency.
                theta_boundary: float
                    Phase difference between lower and upper boundary.
                smearing: bool
                    Return a profile which is smeared out at the edges.

            Returns:
            --------
                xi_lower: float
                    Lower boundary function.
                xi_upper: float
                    Upper boundary function.
        """

        # if variables not supplied set defaults
        if x is None:
            x = self.t
        if eps is None and delta is None:
            eps, delta = self.get_cycle_parameters(x)
        if L is None:
            L = self.L
        if d is None:
            d = self.d
        if kr is None:
            kr = self.kr
        if theta_boundary is None:
            theta_boundary = self.theta_boundary

        # reverse x-coordinate for backward propagation
        if self.loop_direction == '+':
            x = x[::-1]

        def fermi(x, sigma=1):
            """Return the Fermi-Dirac distribution."""
            return 1./(np.exp(-x/sigma) + 1.)

        xi_lower = eps*np.sin((kr + delta)*x)
        xi_upper = d + eps*np.sin((kr + delta)*x + theta_boundary)

        if smearing:
            kr = (self.N - np.sqrt(self.N**2 - 1))*pi
            lambda0 = abs(pi/(kr + delta))
            s = 1./(2*lambda0)
            pre = fermi(x - 3*lambda0, s)*fermi(L - x - 3*lambda0, s)
            return pre*xi_lower, pre*(xi_upper - d) + d
        else:
            return xi_lower, xi_upper

    # def draw_wavefunction(self, instantaneous_eigenbasis=False, save_plot=None):
    #     """Plot wavefunction."""
    #
    #     if instantaneous_eigenbasis:
    #         x, b0, b1 = self.t, self.eVecs_r[:,0], self.eVecs_r[:,1]
    #     else:
    #         x, b0, b1 = self.t, self.phi_a, self.phi_b
    #
    #     yN = len(x)/self.T
    #     y = np.linspace(-0.1,self.d+0.1,yN)
    #
    #     def phi(x,y):
    #         phi = b0 + b1 * (np.sqrt(2.*self.k0/self.k1) *
    #                           np.cos(pi*y)*np.exp(-1j*self.kr*x))
    #         return phi
    #
    #     X, Y = np.meshgrid(x,y)
    #     Z = abs(phi(X,Y))**2
    #
    #     p = plt.pcolormesh(X,Y,Z)
    #     if save_plot:
    #         print save_plot
    #         plt.savefig(save_plot)
    #     #cb = plt.colorbar(p)
    #     #cb.set_label("Wavefunction")
    #
    # def draw_dissipation_coefficient(self, cax=None):
    #     """Plot position dependent dissipation coefficient."""
    #
    #     x, b0, b1 = self.t, self.phi_a, self.phi_b
    #     y = np.linspace(-0.1,self.d+0.1,2)
    #
    #     X, Y = np.meshgrid(x,y)
    #     Z = self.eta_x(X)
    #
    #     bmap = brew.get_map('YlOrRd',
    #                         'sequential', 9).mpl_colormap
    #     p = plt.pcolormesh(X,Y,Z)
    #     p.cmap = bmap
    #     cb = plt.colorbar(p, ax=cax)
    #     cb.set_label("Loss")
    #
    # def get_boundary_contour(self, X, Y):
    #     """Get the boundary contour."""
    #
    #     lower, upper = self.get_boundary(X)
    #     mask_upper = Y > upper
    #     mask_lower = Y < lower
    #     Z = mask_upper + mask_lower
    #
    #     return X, Y, Z
    #
    # def draw_boundary(self):
    #     """Draw the boundary profile."""
    #
    #     x = self.t  #self.t[::2]
    #     #eps, delta = self.get_cycle_parameters(x)
    #
    #     yN = len(x)/self.T
    #     y = np.linspace(-0.1, self.d+0.1, yN)
    #
    #     X, Y = np.meshgrid(x, y)
    #     X, Y, Z = self.get_boundary_contour(X, Y)
    #
    #     plt.contourf(X, Y, Z, [0.9,1], colors="k")
    #     return X, Y, Z


class Neumann(Waveguide):
    """Neumann class."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Neumann boundary
        conditons.

        Copies methods and variables from the Waveguide class.
        """
        Waveguide.__init__(self, **waveguide_kwargs)

        k0, k1 = [ self.k(n) for n in 0, 1 ]
        self.k0, self.k1 = k0, k1
        kr = k0 - k1
        self.kr = kr

        self.x_EP = self.eta / (2.*np.sqrt(k0*k1 * (1.+np.cos(self.theta))))
        self.y_EP = 0.0

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        B = (-1j * (np.exp(1j*self.theta_boundary) + 1) *
                    self.kr/2. * np.sqrt(self.k0/(2.*self.k1)))

        H11 = -self.k0 - 1j*self.eta/2.
        H12 = B*eps
        H21 = B.conj()*eps
        H22 = -self.k0 - delta - 1j*self.eta*self.k0/(2.*self.k1)

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H


class Dirichlet(Waveguide):
    """Dirichlet class."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons.

        Copies methods and variables from the Waveguide class.
       """
        Waveguide.__init__(self, **waveguide_kwargs)
        k0, k1 = [ self.k(n) for n in 1, 2 ]
        self.k0, self.k1 = k0, k1
        kr = k0 - k1
        self.kr = kr

        self.x_EP = self.eta*self.kF*kr*self.d**2/(4*np.pi**2 *
                                        np.sqrt(2*k0*k1*(1.+np.cos(self.theta))))
        self.y_EP = 0.0

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        B = (-1j * (np.exp(1j*self.theta_boundary) + 1) * np.pi**2 /
                self.d**3 / np.sqrt(self.k0*self.k1))
        self.B = B

        H11 = -self.k0 - 1j*self.eta/2.*self.kF/self.k0
        H12 = B*eps
        H21 = B.conj()*eps
        H22 = -self.k0 - delta - 1j*self.eta/2.*self.kF/self.k1

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H

    def get_nodes(self):
        """Return the nodes of the Bloch-eigenvector."""

        k = self.k
        kr = self.kr
        d = self.d

        if not self.loop_type == 'Constant':
            raise Exception("Error: loop_type not 'Constant'!")

        _, b1, b2 = self.solve_ODE()

        x0 = lambda s: (2.*pi/kr * (1-s)/2 - 1j/kr *
                         np.log(s*b1*b2.conj() / (abs(b1)*abs(b2))))
        y0 = lambda s: d/pi*np.arccos(s*0.5*np.sqrt(k(2)/k(1)*abs(b1/b2)))

        xn = [ x0[n] for n in (1, -1) ]
        yn = [ y0[n] for n in (1, -1) ]

        return zip(xn, yn)


class DirichletPositionDependentLoss(Waveguide):
    """Dirichlet class with position dependent loss."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons and position dependent losses.

        Copies methods and variables from the Waveguide class.
        """
        Waveguide.__init__(self, **waveguide_kwargs)
        self.Dirichlet = Dirichlet(**waveguide_kwargs)

    def _get_EP_coordinates(self):
        Gamma = Loss(k=self.k, kF=self.kF, kr=self.kr, d=self.d)
        G1, G2 = [ Gamma.get_Gamma_tilde(x0, y0)
                        for (x0, y0) in self.Dirichlet.get_nodes() ]
        G = G1 + G2
        self.Gamma_tilde = G

        # here B without loss
        kF = self.kF
        kr = self.kr
        B = self.Dirichlet.B

        sq1 = (G[1,1] - kF*G[2,2])**2 + 4.*kF**2*G[1,2]*G[2,1]
        sq2 = (abs(B)**2 + (kF**2*(B*G[2,1]+B.conj()*G[1,2])**2 / (G[1,1]-kF*G[2,2])**2))
        self.x_EP = np.sqrt(sq1)/(2.*np.sqrt(sq2)) * self.eta
        self.y_EP = -2.*kF*(B*G[2,1]+B.conj()*G[1,2])/(G[1,1]-kF*G[2,2]) * self.x_EP

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        B = self.Dirichlet.B

        H11 = -self.k0 - 1j*self.eta*self.Gamma_tilde[1,1]
        H12 = B*eps - 1j*self.eta*self.Gamma_tilde[1,2]
        H21 = B.conj()*eps - 1j*self.eta*self.Gamma_tilde[2,1]
        H22 = -self.k0 - delta - 1j*self.eta*self.Gamma_tilde[2,2]

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H


def plot_figures(show=False, L=100., eta=0.1, N=1.05, phase=-0.1,
                 direction="-", x_EP=0.05):

    import brewer2mpl
    cmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
    colors = cmap.mpl_colors

    params = { "L": L,
               "N": N,
               "eta": eta,
               "init_phase": phase,
               "loop_direction": direction,
               "init_state": "c",
               "calc_adiabatic_state": True,
               "loop_type": "Bell_small_width"}

    WG = Waveguide(**params)
    WG.x_EP = x_EP
    t, cp, cm = WG.solve_ODE()

    # get adiabatic predictions
    cp_ad, cm_ad = (WG.Psi_adiabatic[:,0],
                    WG.Psi_adiabatic[:,1])
    cp_ad *= abs(cp[0])
    cm_ad *= abs(cm[0])

    f, ax1 = plt.subplots()

    ax1.semilogy(t, np.abs(cp), ls="-", color=colors[2], label=r"$|c_+(t)|^2$")
    ax1.semilogy(t, np.abs(cm), ls="-", color=colors[3], label=r"$|c_-(t)|^2$")
    ax1.semilogy(t, np.abs(cp_ad), ls="--", ms="o", color=colors[2], label=r"$|c_+(t)|^2$")
    ax1.semilogy(t, np.abs(cm_ad), ls="--", ms="o", color=colors[3], label=r"$|c_-(t)|^2$")
    ax1.legend(loc="lower left")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylim(1e-6, 1.5e1)

    eps, delta = WG.get_cycle_parameters(t)

    ax2 = plt.axes([0.65, 0.65, .2, .2])
    k0, k1 = [ pi*np.sqrt(N**2 - n**2) for n in 0, 1 ]
    ax2.plot(eta/(2*np.sqrt(2*k0*k1)), "ko")
    ax2.plot(eps, delta, ls="-", color=colors[0])
    ax2.set_xlim(-0.05, 0.05)
    ax2.set_ylim(-0.5, 0.5)

    R = np.abs(cp/cm)
    # R = np.abs(cm/cp)

    print R[-1], 1./R[-1]

    ax3 = plt.axes([0.35, 0.15, .2, .2])
    ax3.plot(t, R, ls="-", color=colors[0])
    ax3.set_ylim(1e-2, np.max(R))

    if show:
        plt.show()
    else:
        plt.savefig("{}.png".format('wg'))


if __name__ == '__main__':
    print "Warning: is normalization symmetric?"

    import argh
    argh.dispatch_command(plot_figures)
