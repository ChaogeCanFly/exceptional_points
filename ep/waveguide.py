#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from ep.base import Base
from ep.dissipation import Loss
from ep.helpers import c_eig


class Waveguide(Base):
    """Waveguide class."""

    def __init__(self, L=100, W=1.0, eta=0.05, N=1.5, theta=0.0,
                 **base_kwargs):
        """Exceptional Point (EP) waveguide class.

        Copies methods and variables from the Base class and adds new
        parameters.

            Additional parameters:
            ----------------------
                W: float
                    Waveguide width
                eta: float
                    Dissipation coefficient
                N: float
                    Number of open modes
                theta: float
                    Phase difference between upper and lower boundary
        """
        Base.__init__(self, T=L, **base_kwargs)

        self.W = W                      # wire width
        self.L = L                      # wire length
        self.eta = eta                  # dissipation coefficient

        self.theta_boundary = theta     # phase angle between upper
                                        # and lower boundary
        self.N = N                      # number of open modes
        self.k = lambda n: np.sqrt(N**2 - n**2)*np.pi/W
                                        # get wavevector in x-direction
        kF = N*np.pi/W                  # Fermi wavevector
        self.kF = kF

    def H(self, t, x=None, y=None):
        """Hamiltonian H is overwritten by inheriting classes."""
        pass

    def get_cycle_parameters(self, t=None):
        """Return the trajectory coordinates (x(t), y(t)) at time t."""

        if t is None:
            t = self.t

        x_R0, y_R0 = self.x_R0, self.y_R0
        w, phi0 = self.w, self.init_phase
        loop_type = self.loop_type

        if loop_type == "Constant":
            t0 = np.ones_like(t)
            return x_R0*t0, y_R0*t0

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

        elif loop_type == "Bell_smooth":
            sign = -int(self.loop_direction + "1")
            lambda1 = lambda t: x_R0 * (1. - np.cos(w*t))
            # take also sign change in w=2pi/T into account
            lambda2 = lambda t: y_R0 * sign * (sign*w*t/pi - 1) + phi0
            smooth = lambda t: self.L/2*(1+np.tanh(2*t/(1-t**2)))
            return lambda1(smooth(2/self.L*t - 1)), lambda2(smooth(2/self.L*t - 1))

        elif loop_type == "Allen-Eberly":
            sign = -int(self.loop_direction + "1")
            lambda1 = lambda t: x_R0 / np.cosh(2.*w*t - 2.*np.pi)
            # take also sign change in w=2pi/T into account
            lambda2 = lambda t: y_R0 * np.tanh(sign*2.*w*t - 2.*np.pi) + phi0
            return lambda1(t), lambda2(t)

        else:
            raise Exception(("Error: loop_type {}"
                             "does not exist!").format(loop_type))

    def get_boundary(self, x=None, eps=None, delta=None, L=None,
                     W=None, kr=None, theta_boundary=None, smearing=False):
        """Get the boundary function xi as a function of the spatial coordinate x.

            Parameters:
            -----------
                x: ndarray
                    Spatial/temporal coordinate.
                eps: float
                    Boundary roughness strength.
                delta: float
                    Boundary frequency detuning.
                L: float
                    Waveguide length.
                W: float
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
        if W is None:
            W = self.W
        if kr is None:
            kr = self.kr
        if theta_boundary is None:
            theta_boundary = self.theta_boundary

        # reverse x-coordinate for backward propagation
        if self.loop_direction == '+':
            x = x[::-1]

        xi_lower = eps*np.sin((kr + delta)*x)
        xi_upper = W + eps*np.sin((kr + delta)*x + theta_boundary)

        if smearing:
            fermi = lambda x, sigma: 1./(np.exp(-x/sigma))
            lambda0 = abs(pi/(kr + delta))
            s = 1./(2*lambda0)
            pre = fermi(x-3*lambda0, s)*fermi(L-x-3*lambda0, s)
            xi_lower *= pre
            xi_upper = pre*(xi_upper - W) + W

        return xi_lower, xi_upper

    def wavefunction(self):
        pass

    def get_boundary_contour(self, X, Y):
        """Get the boundary contour."""

        lower, upper = self.get_boundary(X)
        mask_upper = Y > upper
        mask_lower = Y < lower
        Z = mask_upper + mask_lower

        return X, Y, Z


class Neumann(Waveguide):
    """Neumann class."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Neumann boundary
        conditons.

        Copies methods and variables from the Waveguide class."""
        Waveguide.__init__(self, **waveguide_kwargs)

        k0, k1 = [ self.k(n) for n in 0, 1 ]
        self.k0, self.k1 = k0, k1
        self.kr = k0 - k1

        if self.x_R0 is None or self.y_R0 is None:
            self.x_R0, self.y_R0 = self._get_EP_coordinates()

    def _get_EP_coordinates(self):
        eta = self.eta
        k0 = self.k0
        k1 = self.k1
        W = self.W
        theta_boundary = self.theta_boundary

        x_EP = eta / (2.*np.sqrt(k0*k1 * (1. + np.cos(theta_boundary))))
        y_EP = 0.0

        return x_EP, y_EP

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

    def wavefunction(self):
        x, b0, b1 = self.t, self.phi_a, self.phi_b
        y = np.linspace(0, self.W, len(x)/self.L)

        X, Y = np.meshgrid(x,y)

        PHI = b0 + b1 * (np.sqrt(2.*self.k0/self.k1) *
                          np.cos(pi*Y)*np.exp(-1j*self.kr*X))
        return X, Y, PHI


class Dirichlet(Waveguide):
    """Dirichlet class."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons.

        Copies methods and variables from the Waveguide class."""
        Waveguide.__init__(self, **waveguide_kwargs)

        k0, k1 = [ self.k(n) for n in 1, 2 ]
        self.k0, self.k1 = k0, k1
        kr = k0 - k1
        self.kr = kr

        B = (-1j * (np.exp(1j*self.theta_boundary) + 1) * np.pi**2 /
                self.W**3 / np.sqrt(self.k0*self.k1))
        self.B = B

        self.x_EP, self.y_EP = self._get_EP_coordinates()

        if self.x_R0 is None or self.y_R0 is None:
            self.x_R0, self.y_R0 = self.x_EP, self.y_EP

    def _get_EP_coordinates(self):
        eta = self.eta
        kF = self.kF
        kr = self.kr
        k0 = self.k0
        k1 = self.k1
        W = self.W
        theta_boundary = self.theta_boundary

        x_EP = eta*kF*kr*W**2/(4*np.pi**2 * np.sqrt(2*k0*k1*(1.+np.cos(theta_boundary))))
        y_EP = 0.0

        return x_EP, y_EP

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        H11 = -self.k0 - 1j*self.eta/2.*self.kF/self.k0
        H12 = self.B*eps
        H21 = self.B.conj()*eps
        H22 = -self.k0 - delta - 1j*self.eta/2.*self.kF/self.k1

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H

    def get_nodes(self, x=None, y=None):
        """Return the nodes of the Bloch-eigenvector."""

        if not self.loop_type == 'Constant':
            raise Exception("Error: loop_type not 'Constant'!")

        k = self.k
        kr = self.kr
        W = self.W

        # get eigenvectors of Hermitian system to find the nodes of one
        # Bloch mode
        _, evec = c_eig(self.H(0, x, y))

        # sort eigenvectors: always take the first one returned by c_eig,
        # change if the imaginary part switches sign
        b1, b2 = evec[0,0], evec[1,0]
        if b1.imag > 0 or b2.imag < 0:
            b1, b2 = evec[0,1], evec[1,1]

        # x0 = lambda s: (2.*pi/kr * (1+s)/2
        #                 - 1j/kr * np.log(s*b1*b2.conj() / (abs(b1)*abs(b2))))
        x0 = lambda s: s*np.pi/(2.*kr)
        y0 = lambda s: W/pi*np.arccos(s*0.5*np.sqrt(k(2)/k(1))*abs(b1/b2))

        xn = np.asarray([ x0(n) for n in (+1,-1) ])
        yn = np.asarray([ y0(n) for n in (-1,+1) ])

        # mark invalid node coordinates with np.nan
        # -> caught in DirichletPositionDependentLoss._get_EP_coordinates where
        # G is set to zero for invalid points
        # if np.any(xn < 0.) or np.any(xn > 2.*pi/kr) :
        #     xn *= np.nan
        # if np.any(yn < 0.) or np.any(yn > W):
        #     yn *= np.nan

        if self.verbose:
            print "evec_x =", b1
            print "evec_y =", b2
            print "node xn", xn
            print "node yn", yn

        return np.asarray(zip(xn, yn))

    def wavefunction(self, evecs=False, with_boundary=False):
        if evecs == 'a':
            b0, b1 = [ self.eVecs_r[:,n,0] for n in 0, 1 ]
        elif evecs == 'b':
            b0, b1 = [ self.eVecs_r[:,n,1] for n in 0, 1 ]
        elif evecs == 'c':
            b0, b1 = [ self.eVecs_r[:,n,0] for n in 0, 1 ]
            b2, b3 = [ self.eVecs_r[:,n,1] for n in 0, 1 ]

            mask = np.logical_or(b0.imag > 0, b1.imag <= 0)
            b0[mask], b1[mask] = b2[mask], b3[mask]
        else:
            b0, b1 = self.phi_a, self.phi_b

        x = self.t
        y = np.linspace(-2.*self.x_R0, self.W + 2*self.x_R0, len(x)/self.L)

        X, Y = np.meshgrid(x,y)

        xi_lower, xi_upper = self.get_boundary(x=X)
        xi_lower *= -1  # greens_code counts from top
        xi_upper = self.W + xi_lower  # greens_code counts from top
        if with_boundary:
            Y -= xi_lower
        PHI = (b0 * np.sin(pi/self.W*Y) +
                b1 * np.sqrt(self.k0/self.k1) *
                  np.sin(2*np.pi/self.W*Y)*np.exp(-1j*self.kr*X))
        if with_boundary:
            Y += xi_lower
            PHI[np.logical_or(Y > xi_upper, Y < xi_lower)] = np.nan
        else:
            PHI[np.logical_or(Y > self.W, Y < 0)] = np.nan
        return X, Y, PHI


class DirichletPositionDependentLoss(Dirichlet):
    """Dirichlet class with position dependent loss."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons and position dependent losses.

        Copies methods and variables from the Dirichlet class."""
        dirichlet_kwargs = waveguide_kwargs.copy()
        dirichlet_kwargs.update({'loop_type': 'Constant',
                                 'eta': 0.0})
        self.Dirichlet = Dirichlet(**dirichlet_kwargs)
        Dirichlet.__init__(self, **waveguide_kwargs)

    def _get_EP_coordinates(self):
        Gamma = Loss(k=self.k, kF=self.kF, kr=self.kr, W=self.W)
        self.nodes = self.Dirichlet.get_nodes()

        if np.any(np.isnan(self.nodes)):
            G = np.zeros((2,2))
        else:
            G1, G2 = [ Gamma.get_Gamma_tilde(x0, y0) for (x0, y0) in self.nodes ]
            G = G1 + G2
        self.Gamma_tilde = G

        if self.verbose:
            print "G\n", G

        # here B without loss
        kF = self.kF
        B = self.Dirichlet.B

        sq1 = (G[0,0] - kF*G[1,1])**2 + 4.*kF**2*G[0,1]*G[1,0]
        sq2 = (abs(B)**2 + (kF**2*(B*G[1,0]+B.conj()*G[0,1])**2 / (G[0,0]-kF*G[1,1])**2))

        x_EP = np.sqrt(sq1)/(2.*np.sqrt(sq2)) * self.eta
        y_EP = -2.*kF*(B*G[1,0]+B.conj()*G[0,1])/(G[0,0]-kF*G[1,1]) * x_EP

        return x_EP, y_EP

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        # force re-evaluation of Gamma_tilde
        self.Dirichlet.x_R0 = eps
        self.Dirichlet.y_R0 = delta
        self._get_EP_coordinates()

        B = self.Dirichlet.B

        # damping coefficient
        eps0 = 0.02
        G = 0.5 * (np.sign(eps-eps0) + 1.) * (eps-eps0)**2 * self.Gamma_tilde
        self.Gamma_tilde = G

        H11 = -self.k0 - 1j*self.eta*self.Gamma_tilde[0,0]
        H12 = B*eps - 1j*self.eta*self.Gamma_tilde[0,1]
        H21 = B.conj()*eps - 1j*self.eta*self.Gamma_tilde[1,0]
        H22 = -self.k0 - delta - 1j*self.eta*self.Gamma_tilde[1,1]

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)

        if self.verbose:
            print "t", t
            print "eps", eps
            print "delta", delta
            print "H\n", H
            print "nodes", self.nodes
            print "Gamma_tilde\n", self.Gamma_tilde

        return H

    def get_nodes_waveguide(self, x=None):
        """Return the nodes of the Bloch-eigenvector in the full waveguide."""

        if x is None:
            x = np.arange(0, self.L, 2.*np.pi/self.kr)

        eps, delta = self.get_cycle_parameters(x)
        # L = 2.*np.pi/(self.kr + np.zeros_like(delta))
        L = 2.*np.pi/(self.kr + delta)
        L_sum = np.cumsum(L)
        L_sum -= L_sum[0]

        xnodes, ynodes = [], []
        for epsn, deltan, xn, Ln_sum in zip(eps, delta, x, L_sum):
            Ln = 2.*np.pi/(self.kr + deltan)
            wgn_kwargs = {'N': self.N,
                          'loop_direction': self.loop_direction,
                          'loop_type': 'Constant',
                          'init_state': self.init_state,
                          'eta': 0.0,
                          'L': Ln,
                          'x_R0': epsn,
                          'y_R0': deltan}
            WGn = Dirichlet(**wgn_kwargs)
            nodes = WGn.get_nodes(x=epsn, y=deltan)
            # xnodes.append(nodes[:,0] + xn)
            # xnodes.append(nodes[:,0]/(1+deltan/self.kr) + xn)
            xnodes.append(nodes[:,0]/(1+deltan/self.kr) + Ln_sum)
            ynodes.append(nodes[:,1])

        xnodes, ynodes = [ np.asarray(v).flatten() for v in xnodes, ynodes ]

        return xnodes, ynodes


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
               "loop_type": "Bell"}

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
