#!/usr/bin/env python2.7

import numpy as np
from numpy import pi
from scipy.special import erf, erfc


class Loss(object):
    """Position dependent loss class.

    Returns a function which places Gaussian dissipation profiles at
    coordinates {xn,yn}.
    """

    def __init__(self, xn=0, yn=0, sigmax=0, sigmay=0, WG=None):
        self.xn = xn
        self.yn = yn
        self.sigmax = sigmax
        self.sigmay = sigmay
        if WG:
            self.WG = WG
        else:
            raise Exception("Error: no ep.waveguide.Waveguide object found!")

    def _get_integrals(self, n, m):
        k = self.WG.k
        kF = self.WG.kF
        kr = self.WG.kr
        d = self.WG.d
        T0 = 2.*pi/self.WG.kr

        xn = self.xn
        yn = self.yn

        sigmax = self.sigmax
        sigmay = self.sigmay

        expargx = 0.5*(k(n)-k(m))*(2j*xn - (k(n)-k(m)) * sigmax**2)
        argx1 = (T0 - xn - 1j*(k(n)-k(m))*sigmax**2)/(np.sqrt(2)*sigmax)
        argx2 = (- xn - 1j*(k(n)-k(m))*sigmax**2)/(np.sqrt(2)*sigmax)

        Ix = 0.5*np.exp(expargx) * (erf(argx1) - erf(argx2))

        exparg0 = -(n+m)*pi*(2j*d*yn+(n+m)*pi*sigmay**2)/(2*d**2)
        exparg1 =    2*m*pi*(1j*d*yn+n*pi*sigmay**2)/d**2
        exparg2 =    2*n*pi*(1j*d*yn+m*pi*sigmay**2)/d**2
        exparg3 = 2j*(n+m)*pi*yn/d

        argy1 = (d - yn - 1j*(m-n)*pi*sigmay**2/d)
        argy2 = (    yn + 1j*(m-n)*pi*sigmay**2/d)
        argy3 = (d - yn + 1j*(m-n)*pi*sigmay**2/d)
        argy4 = (    yn - 1j*(m-n)*pi*sigmay**2/d)
        argy5 = (d - yn - 1j*(m+n)*pi*sigmay**2/d)
        argy6 = (    yn + 1j*(m+n)*pi*sigmay**2/d)
        argy7 = (d - yn + 1j*(m+n)*pi*sigmay**2/d)
        argy8 = (    yn - 1j*(m+n)*pi*sigmay**2/d)

        argy = [argy1, argy2, argy3, argy4, argy5, argy6, argy7, argy8]
        argy = [ a/(np.sqrt(2.)*sigmay) for a in argy ]

        Iy = 0.125*np.exp(exparg0) * (-2. +
                np.exp(exparg1) * (erf(argy1) + erf(argy2)) +
                np.exp(exparg2) * (erf(argy3) + erf(argy4)) -
                np.exp(exparg3) * (erf(argy5) + erf(argy6)) +
                erfc(argy7) + erfc(argy8))

        Gamma = 0.5/(pi*d) * kF * kr / np.sqrt(k(n)*k(m)) * (Ix + Iy)

        return Gamma
