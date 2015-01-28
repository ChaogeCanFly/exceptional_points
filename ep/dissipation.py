#!/usr/bin/env python2.7

from __future__ import division
import numpy as np
from numpy import pi
# from scipy.special import erf, erfc


class Loss(object):
    """Position dependent loss class.

    Returns a function which places Gaussian dissipation profiles at
    coordinates {x0,y0}.
    """

    def __init__(self, k=None, kF=None, kr=None, d=None,
                 sigmax=0.05, sigmay=0.05, integrate_R2=True,
                 test_integrals=False):
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.test_integrals = test_integrals
        self.integrate_R2 = integrate_R2

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

        if self.integrate_R2:
            Ix = np.exp(1)**((1/2)*(k(m)+(-1)*k(n))*((1j*2)*x0+sigmax**2*((-1)*k(m)+k(n))))
            Iy = (1/4)*np.exp(1)**((-1/2)*d**(-2)*(m+n)*np.pi*((m+n) \
                *np.pi*sigmay**2+(1j*2)*d*y0))*((-1)+(-1)*np.exp(1)**((1j*2)*d**( \
                -1)*(m+n)*np.pi*y0)+np.exp(1)**(2*d**(-2)*n*np.pi*( \
                m*np.pi*sigmay**2+1j*d*y0))+np.exp(1)**(2*d**(-2)*m*np.pi*( \
                n*np.pi*sigmay**2+1j*d*y0)))
        else:
            # x integration
            expargx = -0.5*(k(n)-k(m))*(2j*x0 + (k(n)-k(m)) * sigmax**2)
            argx1 = (T0 - x0 + 1j*(k(n)-k(m))*sigmax**2)
            argx2 = (   - x0 + 1j*(k(n)-k(m))*sigmax**2)

            # TODO: why do we need a complex() cast here?
            argx = [argx1, argx2]
            argx1, argx2 = [ complex(a)/(np.sqrt(2.)*sigmax) for a in argx ]

            Ix = np.exp(expargx) * np.sqrt(np.pi/2.) * sigmax *  (erf(argx1) - erf(argx2))

            if self.test_integrals:
                Ixalt = np.exp(1)**((1/2)*(k(m)+(-1)*k(n))*(((-1)*k(m)+k(n))*sigmax**2+( \
                        1j*2)*x0))*((1/2)*np.pi)**(1/2)*sigmax*((-1)*erf(2**(-1/2) \
                        *sigmax**(-1)*((1j*(-1))*(k(m)+(-1)*k(n))*sigmax**2+(-1)*x0))+erf( \
                        2**(-1/2)*sigmax**(-1)*(2*np.pi/kr+(1j*(-1))*(k(m)+(-1)*k(n)) \
                        *sigmax**2+(-1)*x0)))

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

            Iy = 0.25*np.exp(expargy0) * np.sqrt(np.pi/2.) * sigmay * (-2. +
                    np.exp(expargy1) * (erf(argy1) + erf(argy2)) +
                    np.exp(expargy2) * (erf(argy3) + erf(argy4)) -
                    np.exp(expargy3) * (erf(argy5) + erf(argy6)) +
                    erfc(argy7) + erfc(argy8))

            if self.test_integrals:
                Iyalt = (1/4)*np.exp(1)**((-1)*d**(-2)*np.pi*((m**2+n**2)*np.pi*sigmay**2+ \
                        (1j*2)*d*m*y0))*((1/2)*np.pi)**(1/2)*sigmay*(np.exp(1)**((1/2) \
                        *d**(-2)*np.pi*((m+n)**2*np.pi*sigmay**2+(1j*2)*d*(3*m+(-1)*n)*y0) \
                        )*erf(2**(-1/2)*sigmay**(-1)*(d+(1j*(-1))*d**(-1)*(m+(-1)*n) \
                        *np.pi*sigmay**2+(-1)*y0))+np.exp(1)**((1/2)*d**(-2)*(m+n)*np.pi*( \
                        (m+n)*np.pi*sigmay**2+(1j*2)*d*y0))*erf(2**(-1/2)*sigmay**(-1)*(d+ \
                        1j*d**(-1)*(m+(-1)*n)*np.pi*sigmay**2+(-1)*y0))+(-1)*np.exp(1)**(( \
                        1/2)*d**(-2)*np.pi*((m+(-1)*n)**2*np.pi*sigmay**2+(1j*2)*d*(3*m+n) \
                        *y0))*erf(2**(-1/2)*sigmay**(-1)*(d+(1j*(-1))*d**(-1)*(m+n) \
                        *np.pi*sigmay**2+(-1)*y0))+(-1)*np.exp(1)**((1/2)*d**(-2)*(m+(-1) \
                        *n)*np.pi*((m+(-1)*n)*np.pi*sigmay**2+(1j*2)*d*y0))*erf(2**(-1/2) \
                        *sigmay**(-1)*(d+1j*d**(-1)*(m+n)*np.pi*sigmay**2+(-1)*y0))+ \
                        np.exp(1)**((1/2)*d**(-2)*(m+n)*np.pi*((m+n)*np.pi*sigmay**2+( \
                        1j*2)*d*y0))*erf(2**(-1/2)*sigmay**(-1)*((1j*(-1))*d**(-1)*(m+(-1) \
                        *n)*np.pi*sigmay**2+y0))+np.exp(1)**((1/2)*d**(-2)*np.pi*((m+n) \
                        **2*np.pi*sigmay**2+(1j*2)*d*(3*m+(-1)*n)*y0))*erf(2**(-1/2) \
                        *sigmay**(-1)*(1j*d**(-1)*(m+(-1)*n)*np.pi*sigmay**2+y0))+(-1) \
                        *np.exp(1)**((1/2)*d**(-2)*(m+(-1)*n)*np.pi*((m+(-1)*n) \
                        *np.pi*sigmay**2+(1j*2)*d*y0))*erf(2**(-1/2)*sigmay**(-1)*((1j*( \
                        -1))*d**(-1)*(m+n)*np.pi*sigmay**2+y0))+(-1)*np.exp(1)**((1/2) \
                        *d**(-2)*np.pi*((m+(-1)*n)**2*np.pi*sigmay**2+(1j*2)*d*(3*m+n)*y0) \
                        )*erf(2**(-1/2)*sigmay**(-1)*(1j*d**(-1)*(m+n)*np.pi*sigmay**2+y0) \
                        ))

                print "Ix, Ixalt, |Ix-Ixalt|", Ix, Ixalt, abs(Ix-Ixalt)
                print "Iy, Iyalt, |Iy-Iyalt|", Iy, Iyalt, abs(Iy-Iyalt)

        Gamma = 1./(2.*np.pi*d) * kF * kr / np.sqrt(k(n)*k(m)) * (Ix * Iy)

        return Gamma

    def get_Gamma_tilde(self, x0, y0):
        """Return the Gamma_tilde matrix."""

        Gamma_tilde = [ self.Gamma(n, m, x0=x0, y0=y0) for n in (1, 2)
                                                        for m in (1, 2) ]
        return np.asarray(Gamma_tilde).reshape(2,-1)


if __name__ == '__main__':
    N = 2.5
    d = 1
    kF = N*np.pi/d
    k = lambda n: np.sqrt(kF**2 - (n*np.pi/d)**2)
    kr = k(1)-k(2)

    L = Loss(k=k, kF=kF, kr=kr, d=d, test_integrals=True)
    print L.Gamma(1,1)
    print L.Gamma(1,2, 0.0923, 0.1231)
    print L.Gamma(2,1, 0.23, 0.31)
    print L.Gamma(2,2, 0.923, 0.1231)
