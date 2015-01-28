#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.waveguide import Dirichlet


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
            sigma_y: float
                Smoothing width of the potential in y-direction.
            width: float
                Waveguide width.
            shape: str
                Potential type.

        Attributes:
        -----------
            imag: (Nx,Ny) ndarray
            real: (Nx,Ny) ndarray
            imag_vector: (Nx*Ny,1) ndarray
            real_vector: (Nx*Ny,1) ndarray
            X, Y: (Nx,Ny) ndarray
    """

    def __init__(self, N=2.5, pphw=20, amplitude=1.0, sigma_y=1e-2,
                 width=1., shape='science'):
        self.N = N
        self.pphw = pphw
        self.amplitude = amplitude
        self.sigma_y = sigma_y
        self.shape = shape
        self.width = width

        self._get_parameters()
        self.imag = self._get_imag_potential()
        self.imag_vector = self._array_to_vector(self.imag)
        self.real = self._get_real_potential()
        self.real_vector = self._array_to_vector(self.real)

    def _get_parameters(self):
        """Return the waveguide parameters for a given number of open modes N."""

        nyout = self.pphw*self.N
        ny = np.floor(self.width*(nyout+1))

        k0, k1 = [ np.sqrt(self.N**2 - (n/self.width)**2)*np.pi for n in 1, 2 ]
        self.kF = k0
        self.kr = k0 - k1
        self.L = 4.*2*np.pi/self.kr

        print vars(self)

        wg_kwargs = {'N': self.N,
                     'L': self.L,
                     'loop_type': 'Constant',
                     'x_R0': 0.05}
        WG = Dirichlet(**wg_kwargs)
        WG.x_R0 = 0.05
        #self.eta_x = WG.eta_x # TODO: implement eta_x

        x = WG.t
        y = np.linspace(0.0, self.width, ny)
        self.X, self.Y = np.meshgrid(x, y)
        self.X0 = np.ones_like(self.X)*self.width/2.

        print "T:", WG.T
        print "eta:", WG.eta
        print "nx:", len(WG.t)
        print "ny:", len(y)

    def _get_imag_potential(self):
        """Return a complex potential."""
        X, Y = self.X, self.Y
        X0 = self.X0
        sigma_y = self.sigma_y
        amplitude = self.amplitude

        if self.shape == 'science':
            imag = np.sin(self.kr*(X - X0))
            imag[Y > Y.mean()] = 0.
            imag[imag < 0.] = 0.
            imag *= -self.kF/2. * amplitude
        elif self.shape == 'smooth':
            imag = self.eta_x(X) * (np.exp(-(Y-0.5)**2/(2*sigma_y**2)) /
                                       np.sqrt(2*np.pi*sigma_y**2))
            imag *= (1-np.cos(np.pi/self.L*X))
            imag *= -self.kF/2. * amplitude
        else:
            imag = np.ones_like(X)
            imag *= -self.kF/2. * amplitude

        return imag

    def _get_real_potential(self):
        """Return a real potential."""
        X, Y = self.X, self.Y
        X0 = self.X0
        sigma_y = self.sigma_y
        amplitude = self.amplitude

        if self.shape == 'science':
            real = np.sin(self.kr*(X - X0) - np.pi/2.)
            real[Y < Y.mean()] = 0.
            real[np.sin(self.kr*(X - X0)) < 0.] = 0.
            real *= self.kF/2. * amplitude
        else:
            real = np.zeros_like(X)

        return real

    def _array_to_vector(self, Z):
        """Turn a NxN matrix into a N*Nx1 array."""

        return Z.flatten(order='F')


def write_potential(N=2.5, pphw=20, amplitude=0.1, sigma_y=1e-2,
                    width=1., shape='science', plot=True):

    p = Potential(N=N, pphw=pphw, amplitude=amplitude, sigma_y=sigma_y,
                  shape=shape)
    imag, imag_vector = p.imag, p.imag_vector
    real, real_vector = p.real, p.real_vector
    X, Y = p.X, p.Y

    if plot:
        plt.pcolormesh(X, Y, imag, cmap='RdBu')
        plt.savefig("imag.png")
        plt.pcolormesh(X, Y, real, cmap='RdBu')
        plt.savefig("real.png")

    np.savetxt("potential_imag.dat", zip(range(len(imag_vector)), imag_vector),
               fmt=["%i", "%.12f"])
    np.savetxt("potential_real.dat", zip(range(len(real_vector)), real_vector),
               fmt=["%i", "%.12f"])


if __name__ == '__main__':
    argh.dispatch_command(write_potential)
