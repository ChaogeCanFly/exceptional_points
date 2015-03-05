#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.waveguide import Dirichlet, DirichletPositionDependentLoss


def gauss(z, mu, sigma):
    return np.exp(-(z-mu)**2 / (2*sigma**2))/(np.sqrt(2.*np.pi)*sigma)


class Potential(object):
    """Basic class to return a spatially dependent potential.

        Parameters:
        -----------
            N: float
                Number of open modes.
            pphw: int
                Points per half wavelength.
            amplitude: float
                Potential strength.
            sigmax: float
                Smoothing width of the potential in x-direction.
            sigmay: float
                Smoothing width of the potential in y-direction.
            W: float
                Waveguide W.
            x_R0: float
                Waveguide loop parameter.
            y_R0: float
                Waveguide loop parameter.
            init_phase: float
                Initial phase/position in frequency space.
            loop_type: str
                Loop parametrization identifier.
            shape: str
                Potential type.
            direction: str
                Injection direction (left|right).
            with_boundary: bool
                Whether to shift the potential positions with the profile
                boundary.

        Attributes:
        -----------
            imag: (Nx,Ny) ndarray
            real: (Nx,Ny) ndarray
            imag_vector: (Nx*Ny,1) ndarray
            real_vector: (Nx*Ny,1) ndarray
            X, Y: (Nx,Ny) ndarray
    """

    def __init__(self, N=2.5, pphw=20, amplitude=1.0, sigmax=1e-1, sigmay=1e-1,
                 L=100, W=1., x_R0=0.05, y_R0=0.4, init_phase=0.0, loop_type='Bell',
                 shape='RAP', direction='right', boundary_only=False,
                 with_boundary=False):
        self.N = N
        self.pphw = pphw
        self.nx = int(L*(pphw*N+1)/W)
        self.ny = int(pphw*N+1)
        self.amplitude = amplitude
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.shape = shape
        self.L = L
        self.W = W
        self.x_R0 = x_R0
        self.y_R0 = y_R0
        self.init_phase = init_phase
        self.loop_type = loop_type
        self.direction = direction
        self.with_boundary = with_boundary

        self._get_parameters()
        if not boundary_only:
            self.imag = self._get_imag_potential()
            self.real = self._get_real_potential()
            self.imag_vector = self._array_to_vector(self.imag)
            self.real_vector = self._array_to_vector(self.real)

    def _get_parameters(self):
        """Return the waveguide parameters for a given number of open modes N."""

        print vars(self)

        wg_kwargs = {'N': self.N,
                     'L': self.L,
                     'tN': self.nx,
                     'loop_type': self.loop_type,
                     'x_R0': self.x_R0,
                     'y_R0': self.y_R0,
                     'init_phase': self.init_phase}
        self.WG = DirichletPositionDependentLoss(**wg_kwargs)

        self.kF = self.WG.k0
        self.kr = self.WG.kr

        if self.direction == 'left':
            self.sign = -1
        else:
            self.sign = 1

        if self.shape == 'science':
            print "Science system size fixed at 4.5*lambda."
            self.L = 4.5*2*np.pi/self.kr

        x = self.WG.t
        y = np.linspace(0.0, self.W, self.ny)
        self.X, self.Y = np.meshgrid(x, y)

        self.X0 = np.ones_like(self.X)*np.pi/self.kr

        print "L:", self.WG.L
        print "eta:", self.WG.eta
        print "nx:", len(self.WG.t)
        print "ny:", len(y)
        print "2pi/kr:", 2.*np.pi/self.kr

    def _get_imag_potential(self):
        """Return a complex potential."""
        X, Y = self.X, self.Y
        X0 = self.X0
        sigmax = self.sigmax
        sigmay = self.sigmay
        amplitude = self.amplitude
        imag = np.zeros_like(X)

        if self.shape == 'science':
            imag = np.sin(self.sign*self.kr*(X - X0))
            imag[Y > Y.mean()] = 0.
            if self.direction == 'left':
                imag[X < (self.L - 4*2*np.pi/self.kr)] = 0.
                imag[X > (self.L - 2*np.pi/self.kr)] = 0.
            else:
                imag[X > 4*2*np.pi/self.kr] = 0.
                imag[X < 2*np.pi/self.kr] = 0.
            imag[imag < 0.] = 0.
        elif self.shape == 'RAP':
            xnodes, ynodes = self.WG.get_nodes_waveguide()
            if self.with_boundary:
                ynodes += -self.WG.get_boundary(xnodes)[0]

            for (xn, yn) in zip(xnodes, ynodes):
                if np.isfinite(xn) and np.isfinite(yn):
                    # greens_code counts from top to bottom: yn -> W - yn
                    # imag += gauss(X, xn, self.sigmax)*gauss(Y, self.WG.W - yn, self.sigmay)
                    imag += gauss(X, xn, self.sigmax)*gauss(Y, yn, self.sigmay)
        else:
            imag = np.ones_like(X)

        imag *= -self.kF/2. * amplitude

        return imag

    def _get_real_potential(self):
        """Return a real potential."""
        X, Y = self.X, self.Y
        X0 = self.X0
        sigmax = self.sigmax
        sigmay = self.sigmay
        amplitude = self.amplitude

        if self.shape == 'science':
            real = np.sin(self.sign*self.kr*(X - (X0 + np.pi/(2.*self.kr))))
            real[Y < Y.mean()] = 0.
            real[real < 0.] = 0.
            if self.direction == 'left':
                real[X < (self.L - 4*2*np.pi/self.kr - np.pi/(2*self.kr))] = 0.
                real[X > (self.L - 2*np.pi/self.kr - np.pi/(2*self.kr))] = 0.
            else:
                real[X > 4*2*np.pi/self.kr + np.pi/(2*self.kr)] = 0.
                real[X < 2*np.pi/self.kr + np.pi/(2*self.kr)] = 0.
            real *= self.kF/2. * amplitude
        else:
            real = np.zeros_like(X)

        return real

    def _array_to_vector(self, Z):
        """Turn a NxN matrix into a N*Nx1 array."""

        return Z.flatten(order='F')


def write_potential(N=2.5, pphw=20, amplitude=1.0, sigmax=1e-1, sigmay=1e-1,
                    L=100., W=1.0, x_R0=0.05, y_R0=0.4, loop_type='Bell',
                    init_phase=0.0, shape='RAP', plot=True,
                    plot_dimensions=False, direction='right',
                    boundary_only=False, with_boundary=False):

    p = Potential(N=N, pphw=pphw, amplitude=amplitude, sigmax=sigmax,
                  sigmay=sigmay, x_R0=x_R0, y_R0=y_R0, init_phase=init_phase,
                  shape=shape, L=L, W=W, loop_type=loop_type,
                  direction=direction, boundary_only=boundary_only,
                  with_boundary=with_boundary)

    if not boundary_only:
        imag, imag_vector = p.imag, p.imag_vector
        real, real_vector = p.real, p.real_vector
    X, Y = p.X, p.Y

    if not boundary_only:
        if plot:
            if plot_dimensions:
                plt.figure(figsize=(L, W))
            plt.pcolormesh(X, Y, imag, cmap='RdBu_r')
            plt.savefig("imag.png")
            plt.pcolormesh(X, Y, real, cmap='RdBu_r')
            plt.savefig("real.png")

        np.savetxt("potential_imag.dat", zip(range(len(imag_vector)), imag_vector),
                   fmt=["%i", "%.12f"])
        np.savetxt("potential_real.dat", zip(range(len(real_vector)), real_vector),
                   fmt=["%i", "%.12f"])
        np.savetxt("potential_imag_xy.dat",
                   zip(X.flatten('F'), Y.flatten('F'), imag_vector, np.fliplr(p.imag).flatten('F')))

    if shape == 'RAP':
        x = p.WG.t
        xi_lower, xi_upper = p.WG.get_boundary(x=x)
        np.savetxt("upper.boundary", zip(range(p.nx), xi_upper))
        np.savetxt("lower.boundary", zip(range(p.nx), xi_lower))


if __name__ == '__main__':
    argh.dispatch_command(write_potential)
