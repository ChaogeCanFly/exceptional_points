#!/usr/bin/env python2.7

import numpy as np
from numpy import pi
from scipy import interpolate, integrate
from scipy.integrate import cumtrapz

from ep.base import Base
from ep.dissipation import Gamma_Gauss
from ep.helpers import c_eig


class Waveguide(Base):
    """Waveguide class."""

    def __init__(self, L=100.0, W=1.0, eta=0.0, N=2.5, theta=0.0, **base_kwargs):
        """Exceptional Point (EP) waveguide class.

        Copies methods and variables from the Base class and adds new
        parameters.

            Additional parameters:
            ----------------------
                L, W: float
                    Waveguide length/width
                N: float
                    Number of open modes
                eta: float
                    Dissipation coefficient
                theta: float
                    Phase difference between upper and lower boundary
        """
        Base.__init__(self, T=L, **base_kwargs)

        self.L = L
        self.W = W
        self.N = N
        self.eta = eta
        self.theta = theta

        self.k = lambda n: np.sqrt(N**2 - n**2)*np.pi/W
        self.kF = N*np.pi/W

    def H(self, t, x=None, y=None, theta=None):
        """Hamiltonian H is overwritten by inheriting classes."""
        pass

    def get_cycle_parameters(self, t=None):
        """Return the trajectory coordinates (x(t), y(t)) at time t."""

        if t is None:
            t = self.t

        L = self.L
        w = self.w
        sign = np.sign(w)
        Delta = self.init_phase

        if self.loop_type == "Constant":
            x, y = [np.ones_like(t)*z for z in self.x_R0, self.y_R0]
        elif self.loop_type == "Landau-Zener":
            x = np.ones_like(t)*self.x_R0
            y = self.y_R0*sign*(sign*w*t/pi - 1.) + Delta
        elif self.loop_type == "Circle":
            x = self.x_R0/2. + self.x_R0/2.*np.cos(w*t + Delta)
            y = self.y_R0 + self.x_R0/2.*np.sin(w*t + Delta)
        elif self.loop_type == "Varcircle":
            x = self.x_R0/2. * (1. - np.cos(w*t))
            y = self.y_R0 - self.x_R0*np.sin(w*t) + Delta
        elif self.loop_type == "Bell":
            # take also sign change in w = 2pi/T into account (in y)
            x = self.x_R0/2. * (1. - np.cos(w*t))
            y = self.y_R0*sign*(sign*w*t/pi - 1.) + Delta
        elif self.loop_type == "Allen-Eberly":
            # mind w = 2pi/L!
            x = self.x_R0 / np.cosh(2.*w*t - 2.*np.pi)
            y = sign*self.y_R0*np.tanh(2.*sign*w*t - 2.*np.pi) + Delta
        # elif self.loop_type == "Allen-Eberly_linearized":
        #     x = self.x_R0 / np.cosh(2.*w*t - 2.*np.pi)
        #     y = (self.kr*t + Delta*t + self.y_R0 * L / (4.*np.pi) *
        #             (np.log(np.cosh(4.*np.pi/L*t - 2.*np.pi)) -
        #                 np.log(np.cosh(2.*np.pi))))
        #     # TODO: take loop_direction into account!
        elif self.loop_type == "Allen-Eberly-Gauss":
            x = self.x_R0 * np.exp(-(t-L/2.)**2 * 2./L)
            y = sign*self.y_R0*np.tanh(2.*sign*w*t - 2.*np.pi) + Delta
        elif self.loop_type == "Bell-Rubbmark":
            x = self.x_R0/2. * (1. - np.cos(w*t))
            y = sign*2.*self.y_R0*(1./(1.+np.exp(-12./L*(t-L/2.)))-0.5) + Delta
        elif self.loop_type == "Allen-Eberly-Rubbmark":
            x = self.x_R0 / np.cosh(2.*w*t - 2.*np.pi)
            y = sign*2.*self.y_R0*(1./(1.+np.exp(-12./L*(t-L/2.)))-0.5) + Delta
        else:
            raise Exception(("Error: loop_type {0}"
                             "does not exist!").format(self.loop_type))
        return x, y

    def get_boundary(self, x=None, eps=None, delta=None, L=None,
                     W=None, kr=None, theta=None, smearing=False,
                     boundary_phase=None):
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
                theta: float
                    Phase difference between lower and upper boundary.
                smearing: bool
                    Return a profile which is smeared out at the edges.
                boundary_phase: float
                    Additional phase of upper and lower boundary.

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
        if theta is None:
            theta = self.theta
        if boundary_phase is None:
            boundary_phase = 0.0

        # reverse x-coordinate for backward propagation
        # corresponds to x -> L - x
        if self.loop_direction == '+':
            x = L - x

        if self.linearized:
            print "Phase linearized!"
            phi = cumtrapz(self.kr + delta,
                           x=self.t, dx=self.dt, initial=0.0)
            xi_lower = eps*np.sin(phi - theta/2.)
            xi_upper = W + eps*np.sin(phi + theta/2.)
        else:
            xi_lower = eps*np.sin((kr + delta)*x + boundary_phase)
            xi_upper = W + eps*np.sin((kr + delta)*x + theta + boundary_phase)

        if smearing:
            def fermi(x, sigma):
                return 1./(1. + np.exp(-x/sigma))
            s = 0.500
            # pre = fermi(x-3.*s, s)*fermi(L-x-3.*s, s)
            pre = fermi(x-4.*s, s)*fermi(L-x-4.*s, s)
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


class Dirichlet(Waveguide):
    """Dirichlet class."""

    def __init__(self, tqd=False, linearized=False, switch_losses_on_off=False,
                 eta0=0.0, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons.

        Copies methods and variables from the Waveguide class.

            Parameters:
            -----------
                tqd: bool
                    Whether to use the transitionless quantum driving
                    algorithm.
                linearized: bool
                    Whether to derive the linearized phase (returns phase of
                    sine function in boundary instead of detuning).
                switch_losses_on_off: bool
                    Whether to scale the eta variable with epsilon^2.
        """
        Waveguide.__init__(self, **waveguide_kwargs)

        self.tqd = tqd
        self.linearized = linearized
        self._tqd_already_calculated = False
        self.switch_losses_on_off = switch_losses_on_off
        self.eta0 = eta0

        k0, k1 = [self.k(n) for n in 1, 2]
        self.k0, self.k1 = k0, k1
        kr = k0 - k1
        self.kr = kr

        B = (-1j * (np.exp(1j*self.theta) + 1) * np.pi**2 /
             self.W**3 / np.sqrt(self.k0*self.k1))
        self.B0 = B

        self.x_EP, self.y_EP = self._get_EP_coordinates()

        if self.x_R0 is None or self.y_R0 is None:
            self.x_R0, self.y_R0 = self.x_EP, self.y_EP

    def _get_EP_coordinates(self):
        """Calculate and return the EP coordinates (x_EP, y_EP)."""
        eta = self.eta
        kF = self.kF
        kr = self.kr
        k0 = self.k0
        k1 = self.k1
        W = self.W
        theta = self.theta

        if not self.switch_losses_on_off:
            x_EP = eta*kF*kr*W**2/(4*np.pi**2 *
                                   np.sqrt(2*k0*k1*(1.+np.cos(theta))))
            y_EP = 0.0
        else:
            y_EP = 0.0
            self.x_from_delta = lambda x: self.L*0.5*((x - self.init_phase)/self.y_R0 + 1.)
            # heaviside = lambda x: 0.5*(np.sign(x) + 1.)
            # self.eta_scaling = lambda x: (((0.5*(1. - np.cos(2.*np.pi/self.L*x)))**1 - 0.15) *
            #                               heaviside(0.5*(1. - np.cos(2.*np.pi/self.L*x)) - 0.15))
            self.eta_scaling = lambda x: (0.5*(1. - np.cos(2.*np.pi/self.L*x)))**2
            f = self.eta_scaling(self.x_from_delta(y_EP))
            x_EP = ((eta*f + self.eta0) * kF * 0.25 *
                    np.abs(1./self.k0 - 1./self.k1)/np.abs(self.B0))

        return x_EP, y_EP

    def H(self, t, x=None, y=None):
        """Return the Dirichlet Hamiltoninan.

            Parameters:
            ----------
                t: float
                    Time at which to evaluate the Hamiltonian.
                x, y: float (optional)
                    Parameters for (eps, delta). If None, (eps, delta) are
                    obtained from the get_cycle_parameters method at time t.
        """
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        if self.tqd:
            if not self._tqd_already_calculated:
                self.tqd_arrays = self.get_quantum_driving_parameters()
                self._tqd_already_calculated = True
            idx = (np.abs(self.t - t)).argmin()
            eps, delta, theta = [a[idx] for a in self.tqd_arrays]
        else:
            theta = self.theta

        if self.switch_losses_on_off:
            # eta = self.eta0 + self.eta * (eps/self.x_R0)**2
            # eta = self.eta0 + self.eta * np.sin(np.pi/self.L*t)
            eta = self.eta0 + self.eta * self.eta_scaling(self.x_from_delta(delta))
        else:
            eta = self.eta

        B = self.B0

        H11 = -self.k0 - 1j*eta/2.*self.kF/self.k0
        H12 = B*eps
        H21 = B.conj()*eps
        H22 = -self.k0 - delta - 1j*eta/2.*self.kF/self.k1

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H

    def get_quantum_driving_parameters(self):
        """Return the adapted parameters (eps_prime, delta, theta_prime) to
        obtain adiabatic dynamics for arbitrary length.
        """
        eps, delta = self.get_cycle_parameters()
        eps_dot, delta_dot = [np.gradient(x, self.dt) for x in eps, delta]

        mixing_angle_dot = 2.*np.abs(self.B0)*(delta*eps_dot-delta_dot*eps)
        mixing_angle_dot /= (delta**2 + 4.*np.abs(self.B0)**2*eps**2)
        self.mixing_angle_dot = mixing_angle_dot

        self.mixing_angle = np.arctan(2.*np.abs(self.B0)*eps/delta)
        self.mixing_angle_dot_alt = np.gradient(self.mixing_angle, self.dt)

        theta_prime = -2.*np.arctan2(mixing_angle_dot, (2*np.abs(self.B0)*eps))

        B_prime = (-1j * (np.exp(1j*theta_prime) + 1.) * np.pi**2 /
                   self.W**3 / np.sqrt(self.k0*self.k1))

        eps_prime = np.sqrt(4.*np.abs(self.B0)**2*eps**2 + mixing_angle_dot**2)
        eps_prime /= 2.*np.abs(B_prime)

        # avoid divergencies
        for n in (0, -1):
            eps_prime[n] = 0.0

        self.eps_prime = eps_prime
        self.delta_prime = delta
        self.theta_prime = theta_prime

        return eps_prime, delta, theta_prime

    def get_nodes(self, x=None, y=None, return_evecs=False):
        """Return the nodes of the Bloch-eigenvector in the unit cell."""

        if not self.loop_type == 'Constant':
            raise Exception("Error: loop_type not 'Constant'!")

        k = self.k
        kr = self.kr
        W = self.W

        evals, evecs = c_eig(self.H(0, x, y))

        j0 = 0
        j1 = 1
        b1, b2 = [evecs[i, j0] for i in (0, 1)]
        # if b1.imag > 0 or b2.imag < 0:
        if y <= 0:
            b1, b2 = [evecs[i, j1] for i in (0, 1)]
            evecs[:, 0], evecs[:, 1] = evecs[:, 1], evecs[:, 0]
            evals[0], evals[1] = evals[1], evals[0]

        # # write eigensystem to file
        # with open("evecs_{}_{}.dat".format(self.loop_direction,
        #                                    self.init_state), "a") as f:
        #     ev = evals
        #     e = evecs
        #     data = (ev[0].real, ev[0].imag, ev[1].real, ev[1].imag,
        #             e[0,0].real, e[0,0].imag, e[1,0].real, e[1,0].imag,
        #             e[0,1].real, e[0,1].imag, e[1,1].real, e[1,1].imag)
        #     np.savetxt(f, data, newline="  ", fmt='%.5e')
        #     f.write("\n")

        def x0(s):
            """Return x-coordinates in unit cell.  Only valid for boundary
            phase parameter vartheta = 0."""
            return (2.*pi/kr * (1+s)/2 - 1j/kr *
                    np.log(-s*np.exp(-1j*np.pi/2)*b1*b2.conj() / (abs(b1)*abs(b2))))

        # def x0(s):
        #     """Return x-coordinates in unit cell.  Only valid for boundary
        #     phase parameter vartheta = 0."""
        #     return s*np.pi/(2.*kr) + (1.-s)/2. * 2.*pi/kr

        def y0(s):
            """Return y-coordinates in unit cell."""
            return W/pi*np.arccos(s*0.5*np.sqrt(k(2)/k(1))*abs(b1/b2))

        xn = np.asarray([x0(n) for n in (+1, -1)])
        yn = np.asarray([y0(n) for n in (-1, +1)])

        # # write coordinates to file
        # with open("coords_{}_{}.dat".format(self.loop_direction,
        #                                     self.init_state), "a") as f:
        #     data = (xn[0], yn[0], xn[1], yn[1])
        #     np.savetxt(f, data, newline="  ", fmt='%.5e')
        #     f.write("\n")

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

        if not return_evecs:
            return np.asarray(zip(xn, yn))
        else:
            return evecs

    def wavefunction(self, evecs=False, with_boundary=False):
        """Return the wavefunction Psi(x,y)."""
        if evecs == 'a':
            b0, b1 = [self.eVecs_r[:, n, 0] for n in (0, 1)]
        elif evecs == 'b':
            b0, b1 = [self.eVecs_r[:, n, 1] for n in (0, 1)]
        elif evecs == 'c':
            b0, b1 = [self.eVecs_r[:, n, 0] for n in (0, 1)]
            b2, b3 = [self.eVecs_r[:, n, 1] for n in (0, 1)]

            mask = np.logical_or(b0.imag > 0, b1.imag <= 0)
            b0[mask], b1[mask] = b2[mask], b3[mask]
        else:
            b0, b1 = self.phi_a, self.phi_b

        x = self.t
        y = np.linspace(-2.*self.x_R0, self.W + 2*self.x_R0, len(x)/self.L)

        X, Y = np.meshgrid(x, y)

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


class DirichletReduced(Dirichlet):
    def __init__(self, **dirichlet_kwargs):
        Dirichlet.__init__(self, **dirichlet_kwargs)

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        if self.switch_losses_on_off:
            eta = self.eta0 + self.eta * self.eta_scaling(self.x_from_delta(delta))
        else:
            eta = self.eta

        H11 = delta - 1j*eta/2. * self.kF/self.k0
        H12 = np.abs(self.B0)*eps
        H21 = H12
        H22 = -1j*eta/2. * self.kF/self.k1

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H


class DirichletPositionDependentLoss(Dirichlet):
    """Dirichlet class with position dependent loss."""

    def __init__(self, eta0=0.0, sigma=1e-2, switch_losses_on_off=False,
                 **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons and position dependent losses.

        Copies methods and variables from the Dirichlet class.

            Additional parameters:
            ----------------------
                eta0: float
                    Constant loss strength.
                sigma: float
                    Standard deviation of the Gaussian loss potential.
        """
        Dirichlet.__init__(self, **waveguide_kwargs)
        dirichlet_kwargs = waveguide_kwargs.copy()

        self.eta0 = eta0
        self.sigma = sigma
        self.switch_losses_on_off = switch_losses_on_off
        dirichlet_kwargs.update({'loop_type': 'Constant',
                                 'eta': 0.0})
        self.Dirichlet = Dirichlet(**dirichlet_kwargs)
        self._get_EP_coordinates()

    def _get_loss_matrix(self, x=None, y=None):
        Gamma = Gamma_Gauss(k=self.k, kF=self.kF, kr=self.kr, W=self.W,
                            sigmax=self.sigma, sigmay=self.sigma)
        self.nodes = self.Dirichlet.get_nodes(x=x, y=y)

        if np.any(np.isnan(self.nodes)):
            G = np.zeros((2, 2))
        else:
            G1, G2 = [Gamma.get_matrix(x0, y0) for (x0, y0) in self.nodes]
            G = G1 + G2

        # dyadic product
        # evals, evecs = c_eig(self.Dirichlet.H(0, x, y))
        # # if evals[0] > evals[1]:
        # #     evecs[:, 0], evecs[:, 1] = evecs[:, 1], evecs[:, 0]
        # b1, b2 = [evecs[i, 0] for i in (0, 1)]
        # G = np.outer(b1, b1.conj())
        # if self.loop_direction == '-' and self.init_state == 'b':
        #     G *= 0
        # if self.loop_direction == '+' and self.init_state == 'a':
        #     G *= 0

        if self.verbose:
            print "G\n", G

        return G

    # def _get_EP_coordinates(self, x=None, y=None):
    #     return x, y
        # merge with DirichletNumericPotential?

        # kF = self.kF
        # # TODO: B of Dirichlet necessary here? Why not self.B?
        # B = self.Dirichlet.B0
        #
        # G = self._get_loss_matrix()
        # sq1 = (G[0, 0] - kF*G[1, 1])**2 + 4.*kF**2*G[0, 1]*G[1, 0]
        # sq2 = (abs(B)**2 + (kF**2*(B*G[1, 0] + B.conj()*G[0, 1])**2 /
        #        (G[0, 0] - kF*G[1, 1])**2))
        #
        # x_EP = np.sqrt(sq1)/(2.*np.sqrt(sq2)) * self.eta
        # y_EP = -2.*kF*(B*G[1, 0]+B.conj()*G[0, 1])/(G[0, 0]-kF*G[1, 1]) * x_EP
        #
        # return x_EP, y_EP

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        Gamma_matrix = self._get_loss_matrix(x=eps, y=delta)

        # damping coefficient
        # eps0 = 0.25*self.x_R0
        # envelope = 0.5 * (np.sign(eps-eps0) + 1.) * (eps-eps0)**2
        # envelope = eps**2
        # envelope = (eps/self.x_R0)**2
        #
        # Gamma_matrix *= envelope

        H11 = -self.k0
        H12 = self.B0*eps
        H21 = self.B0.conj()*eps
        H22 = -self.k0 - delta

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)

        Gamma_matrix_const = np.array([[self.kF/self.k0, 0.0],
                                       [0.0, self.kF/self.k1]], dtype=complex)

        if self.switch_losses_on_off:
            # eta = self.eta * (eps/self.x_R0)**2
            eta = self.eta0 + self.eta * self.eta_scaling(self.x_from_delta(delta))
        else:
            eta = self.eta

        H -= 1j*eta/2.*Gamma_matrix
        H -= 1j*self.eta0/2.*Gamma_matrix_const

        if self.verbose:
            print "t", t
            print "eps", eps
            print "delta", delta
            print "H\n", H
            print "nodes", self.nodes
            print "Gamma_matrix\n", Gamma_matrix

        return H

    def get_nodes_waveguide(self, x=None):
        """Return the nodes of the Bloch-eigenvector in the full waveguide."""

        if x is None:
            x = np.arange(0, self.L, 2.*np.pi/self.kr)

        eps, delta = self.get_cycle_parameters(x)
        L = 2.*np.pi/np.abs(self.kr + delta)
        L_sum = L #np.cumsum(L)
        # L_sum -= L_sum[0]

        xnodes, ynodes = [], []
        for epsn, deltan, xn, Ln_sum in zip(eps, delta, x, L_sum):
            print "epsn, deltan, xn, Ln_sum", epsn, deltan, xn, Ln_sum
            Ln = 2.*np.pi/np.abs(self.kr + deltan)
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
            xnodes.append(nodes[:, 0] + Ln_sum)
            ynodes.append(nodes[:, 1])

        xnodes, ynodes = [np.asarray(v).flatten() for v in xnodes, ynodes]

        return xnodes, ynodes


class DirichletPositionDependentLossReduced(DirichletPositionDependentLoss):
    """Dirichlet class with position dependent loss."""

    def __init__(self, eta0=0.0, sigma=1e-2, switch_losses_on_off=False,
                 **waveguide_kwargs):
        DirichletPositionDependentLoss.__init__(self, **waveguide_kwargs)
        dirichlet_kwargs = waveguide_kwargs.copy()

        self.eta0 = eta0
        self.sigma = sigma
        self.switch_losses_on_off = switch_losses_on_off
        dirichlet_kwargs.update({'loop_type': 'Constant',
                                 'eta': 0.0})
        self.Dirichlet = DirichletReduced(**dirichlet_kwargs)
        self._get_EP_coordinates()

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        H11 = delta
        H12 = np.abs(self.B0)*eps
        H21 = np.abs(self.B0)*eps
        H22 = 0.0

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)

        Gamma_matrix_const = np.array([[self.kF/self.k0, 0.0],
                                       [0.0, self.kF/self.k1]], dtype=complex)
        Gamma_matrix = self._get_loss_matrix(x=eps, y=delta)
        Gamma_matrix[0,1] *= np.exp(+1j*np.pi/2.)
        Gamma_matrix[1,0] *= np.exp(-1j*np.pi/2.)

        # if self.switch_losses_on_off:
        #     # eta = self.eta * (eps/self.x_R0)**2
        #     eta = self.eta0 + self.eta * self.eta_scaling(self.x_from_delta(delta))
        # else:
        #     eta = self.eta

        f_delta = 0.5*(1. + np.cos(np.pi*(delta - self.init_phase)/self.y_R0))
        f_eps = eps/self.x_R0
        f_delta, f_eps = [u**2 for u in (f_delta, f_eps)]
        # f_delta, f_eps = [u**1 for u in (f_delta, f_eps)]
        f_diff = f_delta - f_eps

        # coefficient eta0 and eta should be the same (see SI)
        H -= 1j*self.eta0/2. * Gamma_matrix_const * f_diff
        H -= 1j*self.eta/2. * Gamma_matrix * (1. - f_diff) * f_eps
        # the last f_eps is needed to ensure that at final points of the loop
        # the absorption is switched off

        if self.verbose:
            print "t", t
            print "eps", eps
            print "delta", delta
            print "H\n", H
            print "nodes", self.nodes
            print "Gamma_matrix\n", Gamma_matrix

        return H


class DirichletDyadicReduced(DirichletPositionDependentLossReduced):
    """Dirichlet class with position dependent loss."""

    def _get_loss_matrix(self, x=None, y=None):

        # dyadic product
        evals, evecs = c_eig(self.Dirichlet.H(0, x, y))
        v1, v2 = [evecs[:, n] for n in (0, 1)]
        if y >= 0:
            v = v2
        else:
            v = v1
        G = np.outer(v, v.conj())

        if self.verbose:
            print "G\n", G

        return G




class DirichletNumericPotential(Dirichlet):
    """Dirichlet class with potential input."""

    def __init__(self, potential_file=None, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Dirichlet boundary
        conditons and position dependent losses.

        Copies methods and variables from the Dirichlet class.

                Additional parameters:
                ----------------------
                    potential_file: .npz container
                        X, Y and P meshes of the potential and the underlying
                        grid

        """
        dirichlet_kwargs = waveguide_kwargs.copy()
        dirichlet_kwargs.update({'loop_type': 'Constant',
                                 'eta': 0.0})
        self.Dirichlet = Dirichlet(**dirichlet_kwargs)
        Dirichlet.__init__(self, **waveguide_kwargs)

        self.potential_file = potential_file

    def _get_potential_interpolating_function(self):

        F = np.load(self.potential_file)
        xgrid, ygrid, pgrid = [F[s].T for s in 'X', 'Y', 'P']
        x, y = [np.unique(i) for i in xgrid, ygrid]

        return xgrid, ygrid, interpolate.RectBivariateSpline(x, y, pgrid)

    def _get_EP_position(self, tn):
        prefactor = lambda n, m: (1./(2.*pi*self.W) * self.kF * self.kr / np.sqrt(self.k(n)*self.k(m)))

        def Gamma(y, x, n, m, part=np.real):
            G = self.numeric_potential(x, y)
            G *= prefactor(n, m)*np.sin(n*pi/self.W*y)*np.sin(m*pi/self.W*y)
            G *= np.exp(1j*(self.k(n)-self.k(m)*x))
            return part(G)

        G = lambda n, m: (integrate.quad(Gamma, 0, self.W, args=(tn, n, m, np.real))[0] +
                           1j*integrate.quad(Gamma, 0, self.W, args=(tn, n, m, np.imag))[0])

        eps_EP = np.sqrt((G(0, 0) - G(1, 1))**2 + 4.*G(0, 1)*G(1, 0))
        eps_EP /= (2.*np.sqrt(abs(self.B0)**2 + (self.B0.conj()*G(0, 1) +
                    self.B0 * G(1, 0))**2 / (G(0, 0) - G(1, 1)**2)))
        delta_EP = - 2. * (self.B0.conj()*G(0, 1) + self.B0 * G(1, 0)) * eps_EP
        delta_EP /= (G(0, 0) - G(1, 1))
        print "tn", tn, "eps_EP", eps_EP, "delta_EP", delta_EP
        print "G_00", G(0, 0)
        print "G_01", G(0, 1)
        print "G_10", G(1, 0)
        print "G_11", G(1, 1)
        return eps_EP, delta_EP

    def get_EP_positions(self):
        xgrid, ygrid, self.numeric_potential = self._get_potential_interpolating_function()
        # print "xgrid", xgrid.min(), xgrid.max()
        # print "ygrid", ygrid.min(), ygrid.max()
        # x = np.linspace(0, self.L)
        # y = np.linspace(0, self.W)
        # X, Y = np.meshgrid(x, y)
        # z = self.numeric_potential.ev(X, Y)
        # import matplotlib.pyplot as plt
        # plt.clf()
        # plt.pcolormesh(x, y, z)
        # zz = self.numeric_potential.ev(xgrid, ygrid)
        # plt.pcolormesh(xgrid, ygrid, zz)
        # plt.show()
        eps_EP, delta_EP = np.asarray([self._get_EP_position(tN) for tN in self.t]).T

        return eps_EP, delta_EP


class Neumann(Waveguide):
    """Neumann class."""

    def __init__(self, **waveguide_kwargs):
        """Exceptional Point (EP) waveguide class with Neumann boundary
        conditons.

        Copies methods and variables from the Waveguide class."""
        Waveguide.__init__(self, **waveguide_kwargs)

        k0, k1 = [self.k(n) for n in 0, 1]
        self.k0, self.k1 = k0, k1
        self.kr = k0 - k1

        B = (-1j * (np.exp(1j*self.theta) + 1) *
             self.kr/2. * np.sqrt(self.k0/(2.*self.k1)))
        self.B0 = B

        self.x_EP, self.y_EP = self._get_EP_coordinates()

        if self.x_R0 is None or self.y_R0 is None:
            self.x_R0, self.y_R0 = self._get_EP_coordinates()

    def _get_EP_coordinates(self):
        eta = self.eta
        k0 = self.k0
        k1 = self.k1
        theta = self.theta

        x_EP = eta / (2.*np.sqrt(k0*k1 * (1. + np.cos(theta))))
        y_EP = 0.0

        return x_EP, y_EP

    def H(self, t, x=None, y=None):
        if x is None and y is None:
            eps, delta = self.get_cycle_parameters(t)
        else:
            eps, delta = x, y

        B = self.B0

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

        X, Y = np.meshgrid(x, y)

        PHI = b0 + b1 * (np.sqrt(2.*self.k0/self.k1) *
                         np.cos(pi*Y)*np.exp(-1j*self.kr*X))
        return X, Y, PHI


def plot_figures(show=False, L=100., eta=0.1, N=1.05, phase=-0.1,
                 direction="-", x_EP=0.05):

    import matplotlib.pyplot as plt

    import brewer2mpl
    cmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
    colors = cmap.mpl_colors

    params = {"L": L,
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
    cp_ad, cm_ad = (WG.Psi_adiabatic[:, 0],
                    WG.Psi_adiabatic[:, 1])
    cp_ad *= abs(cp[0])
    cm_ad *= abs(cm[0])

    f, ax1 = plt.subplots()

    ax1.semilogy(t, np.abs(cp), ls="-", color=colors[2], label=r"$|c_+(t)|^2$")
    ax1.semilogy(t, np.abs(cm), ls="-", color=colors[3], label=r"$|c_-(t)|^2$")
    ax1.semilogy(t, np.abs(cp_ad), ls="--", ms="o",
                 color=colors[2], label=r"$|c_+(t)|^2$")
    ax1.semilogy(t, np.abs(cm_ad), ls="--", ms="o",
                 color=colors[3], label=r"$|c_-(t)|^2$")
    ax1.legend(loc="lower left")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylim(1e-6, 1.5e1)

    eps, delta = WG.get_cycle_parameters(t)

    ax2 = plt.axes([0.65, 0.65, .2, .2])
    k0, k1 = [np.pi*np.sqrt(N**2 - n**2) for n in 0, 1]
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
