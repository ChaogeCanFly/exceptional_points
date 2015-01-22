#!/usr/bin/env python2.7

import numpy as np
from numpy import pi
import scipy.linalg
from scipy.special import erf, erfc


class Loss(object):
    """Position dependent loss class.

    Returns a function which places Gaussian dissipation profiles at
    coordinates {x0,y0}.
    """

    def __init__(self, x0=0, y0=0, sigmax=0, sigmay=0, WG=None):
        self.x0 = x0
        self.y0 = y0
        self.sigmax = sigmax
        self.sigmay = sigmay
        if WG:
            self.WG = WG
        else:
            raise Exception("Error: no ep.waveguide.Waveguide object found!")

    def _get_nodes(self):
        """Return the nodes of the Bloch-eigenvector."""

        k = self.WG.k
        kr = self.WG.kr
        d = self.WG.d

        if not self.WG.loop_type == 'Constant':
            raise Exception("Error: loop_type not 'Constant'!")

        _, b1, b2 = self.WG.solve_ODE()

        x0 = lambda s: (2.*pi/kr * (1-s)/2 - 1j/kr *
                         np.log(s*b1*b2.conj() / (abs(b1)*abs(b2))))
        y0 = lambda s: d/pi*np.arccos(s*0.5*np.sqrt(k(2)/k(1)*abs(b1/b2)))

        return ((x0(1), y0(1)), (x0(-1), y0(-1)))

    def _get_Gamma_integrals(self, n, m, x0=0, y0=0):
        """Return the integrals needed for Gamma_tilde."""
        k = self.WG.k
        kF = self.WG.kF
        kr = self.WG.kr
        d = self.WG.d
        T0 = 2.*pi/self.WG.kr

        sigmax = self.sigmax
        sigmay = self.sigmay

        # x integration
        expargx = -0.5*(k(n)-k(m))*(2j*x0 + (k(n)-k(m)) * sigmax**2)
        argx1 = (T0 - x0 + 1j*(k(n)-k(m))*sigmax**2)
        argx2 = (   - x0 + 1j*(k(n)-k(m))*sigmax**2)

        argx = [argx1, argx2]
        argx = [ a/(np.sqrt(2.)*sigmax) for a in argx ]

        Ix = 0.5*np.exp(expargx) * (erf(argx1) - erf(argx2))

        # y integration
        exparg0 = -(n+m)*pi*(2j*d*y0+(n+m)*pi*sigmay**2)/(2*d**2)
        exparg1 =    2*m*pi*(1j*d*y0+n*pi*sigmay**2)/d**2
        exparg2 =    2*n*pi*(1j*d*y0+m*pi*sigmay**2)/d**2
        exparg3 = 2j*(n+m)*pi*y0/d

        argy1 = (d - y0 - 1j*(m-n)*pi*sigmay**2/d)
        argy2 = (    y0 + 1j*(m-n)*pi*sigmay**2/d)
        argy3 = (d - y0 + 1j*(m-n)*pi*sigmay**2/d)
        argy4 = (    y0 - 1j*(m-n)*pi*sigmay**2/d)
        argy5 = (d - y0 - 1j*(m+n)*pi*sigmay**2/d)
        argy6 = (    y0 + 1j*(m+n)*pi*sigmay**2/d)
        argy7 = (d - y0 + 1j*(m+n)*pi*sigmay**2/d)
        argy8 = (    y0 - 1j*(m+n)*pi*sigmay**2/d)

        argy = [argy1, argy2, argy3, argy4, argy5, argy6, argy7, argy8]
        argy = [ a/(np.sqrt(2.)*sigmay) for a in argy ]

        Iy = 0.125*np.exp(exparg0) * (-2. +
                np.exp(exparg1) * (erf(argy1) + erf(argy2)) +
                np.exp(exparg2) * (erf(argy3) + erf(argy4)) -
                np.exp(exparg3) * (erf(argy5) + erf(argy6)) +
                erfc(argy7) + erfc(argy8))

        Gamma = 0.5/(pi*d) * kF * kr / np.sqrt(k(n)*k(m)) * (Ix + Iy)

        return Gamma

    def _get_Gamma_tilde(self):
        """Return the Gamma_tilde matrix."""

        Gamma_tilde = [ Gamma(n, m, x0=x0, y0=y0) for n in 1, 2
                                                    for m in 1, 2
                                                      for (x0, y0) in zip(self._get_nodes()) ]

        return Gamma_tilde


