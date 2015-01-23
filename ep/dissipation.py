#!/usr/bin/env python2.7

import numpy as np
from numpy import pi
from scipy.special import erf, erfc


class Loss(object):
    """Position dependent loss class.

    Returns a function which places Gaussian dissipation profiles at
    coordinates {x0,y0}.
    """

    def __init__(self, k=None, kF=None, kr=None, d=None, sigmax=0.1, sigmay=0.1):

        self.sigmax = sigmax
        self.sigmay = sigmay

        if None in (k, kF, kr, d):
            raise Exception("Error: need wavenumber/width information!")
        else:
            self.k = k
            self.kF = kF
            self.kr = kr
            self.d = d

    def Gamma(self, n, m, x0=0, y0=0):
        """Return the integrals needed for Gamma_tilde."""
        k = self.k
        kF = self.kF
        kr = self.kr
        d = self.d
        T0 = 2.*pi/self.kr

        sigmax = self.sigmax
        sigmay = self.sigmay

        # x integration
        expargx = -0.5*(k(n)-k(m))*(2j*x0 + (k(n)-k(m)) * sigmax**2)
        argx1 = (T0 - x0 + 1j*(k(n)-k(m))*sigmax**2)
        argx2 = (   - x0 + 1j*(k(n)-k(m))*sigmax**2)

        # print "x0", x0
        # print "y0", y0
        # print "n", n
        # print "m", m
        # print "expargx", expargx
        # print "e()", np.exp(expargx)
        # print "argx1", argx1
        # print "type argx1", type(argx1)
        # print "cast argx1", complex(argx1)
        # print "type cast argx1", type(complex(argx1))

        # TODO: why do we need a complex() cast here?
        argx = [argx1, argx2]
        argx1, argx2 = [ complex(a)/(np.sqrt(2.)*sigmax) for a in argx ]

        Ix = 0.5*np.exp(expargx) * (erf(argx1) - erf(argx2))

        # y integration
        expargy0 = -(n+m)*pi*(2j*d*y0+(n+m)*pi*sigmay**2)/(2*d**2)
        expargy1 =    2*m*pi*(1j*d*y0+n*pi*sigmay**2)/d**2
        expargy2 =    2*n*pi*(1j*d*y0+m*pi*sigmay**2)/d**2
        expargy3 = 2j*(n+m)*pi*y0/d

        argy1 = (d - y0 - 1j*(m-n)*pi*sigmay**2/d)
        argy2 = (    y0 + 1j*(m-n)*pi*sigmay**2/d)
        argy3 = (d - y0 + 1j*(m-n)*pi*sigmay**2/d)
        argy4 = (    y0 - 1j*(m-n)*pi*sigmay**2/d)
        argy5 = (d - y0 - 1j*(m+n)*pi*sigmay**2/d)
        argy6 = (    y0 + 1j*(m+n)*pi*sigmay**2/d)
        argy7 = (d - y0 + 1j*(m+n)*pi*sigmay**2/d)
        argy8 = (    y0 - 1j*(m+n)*pi*sigmay**2/d)

        argy = [argy1, argy2, argy3, argy4, argy5, argy6, argy7, argy8]
        (argy1, argy2, argy3, argy4,
         argy5, argy6, argy7, argy8) = [ complex(a)/(np.sqrt(2.)*sigmay) for a in argy ]

        Iy = 0.125*np.exp(expargy0) * (-2. +
                np.exp(expargy1) * (erf(argy1) + erf(argy2)) +
                np.exp(expargy2) * (erf(argy3) + erf(argy4)) -
                np.exp(expargy3) * (erf(argy5) + erf(argy6)) +
                erfc(argy7) + erfc(argy8))

        Gamma = 0.5/(pi*d) * kF * kr / np.sqrt(k(n)*k(m)) * (Ix + Iy)

        return Gamma

    def get_Gamma_tilde(self, x0, y0):
        """Return the Gamma_tilde matrix."""

        Gamma_tilde = [ self.Gamma(n, m, x0=x0, y0=y0) for n in (1, 2)
                                                        for m in (1, 2) ]

        return np.asarray(Gamma_tilde).reshape(2,-1)
