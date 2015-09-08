#!/usr/bin/env python2.7

from __future__ import division
import numpy as np
from numpy import pi
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline
from scipy.special import erf, erfc


class Gamma(object):
    """Position dependent loss class."""
    def __init__(self, k=None, kF=None, kr=None, W=None):
        if None in (k, kF, kr, W):
            raise Exception("Error: need wavenumber/width information!")
        else:
            self.k = k
            self.kF = kF
            self.kr = kr
            self.W = W

    def get_matrix_element(self, n, m):
        pass

    def get_matrix(self):
        pass


class Gamma_From_Grid(Gamma):
    """Position dependent loss class which reads a .npz mesh."""

    def __init__(self, x0, potential_file=None, **gamma_kwargs):
        Gamma.__init__(self, **gamma_kwargs)
        self.potential_file = potential_file
        self.x0 = x0
        (self.x, self.y, self.X,
         self.Y, self.P, self.P_interpolate) = self._load_potential()

    def _load_potential(self):
        P_npz = np.load(self.potential_file)
        X, Y, P = [P_npz[s].T for s in 'X', 'Y', 'P']
        x, y = [np.unique(i) for i in X, Y]

        # greens_code counts from top
        P = np.flipud(P)

        return x, y, X, Y, P, RectBivariateSpline(y, x)

    def get_matrix_element(self, n, m):

        def coeff(x, y):
            coeff_x = np.exp(1j*(self.k(n)-self.k(m))*x)
            coeff_y = np.sin(n*np.pi/self.W*y)*np.sin(m*np.pi/self.W*y)
            heaviside = 0.5*(np.sign(self.x[-1] - x) + 1.)
            return coeff_x*coeff_y*heaviside

        integral = dblquad(lambda x, y: self.P_interpolate(y, x)*coeff(x, y),
                           self.x0, self.x0 + 2.*np.pi/self.kr,
                           lambda x: 0, lambda x: self.W)

        prefactor = 1./(np.pi*self.W) * self.kF * self.kr
        prefactor /= np.sqrt(self.k(n)*self.k(m))

        return prefactor*integral

    def get_matrix(self):
        pass


class Gamma_Gauss(Gamma):
    """Position dependent loss class.

    Returns a function which places Gaussian dissipation profiles at
    coordinates {x0,y0}.
    """

    def __init__(self, sigmax=1.e-2, sigmay=1.e-2, integrate_R2=True,
                 test_integrals=False, **gamma_kwargs):
        Gamma.__init__(self, **gamma_kwargs)
        self.sigmax = sigmax
        self.sigmay = sigmay

        self.test_integrals = test_integrals
        self.integrate_R2 = integrate_R2

    def get_matrix_element(self, n, m, x0=0, y0=0):
        k = self.k
        kF = self.kF
        kr = self.kr
        W = self.W
        T0 = 2.*pi/self.kr

        sigmax = self.sigmax
        sigmay = self.sigmay

        if self.integrate_R2:
            Ix = np.exp(1)**((1/2)*(k(m)+(-1)*k(n))*((1j*2)*x0+sigmax**2*((-1)*k(m)+k(n))))
            Iy = (1/4)*np.exp(1)**((-1/2)*W**(-2)*(m+n)*np.pi*((m+n) \
                *np.pi*sigmay**2+(1j*2)*W*y0))*((-1)+(-1)*np.exp(1)**((1j*2)*W**( \
                -1)*(m+n)*np.pi*y0)+np.exp(1)**(2*W**(-2)*n*np.pi*( \
                m*np.pi*sigmay**2+1j*W*y0))+np.exp(1)**(2*W**(-2)*m*np.pi*( \
                n*np.pi*sigmay**2+1j*W*y0)))
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
            expargy0 = -(n+m)*pi*(2j*W*y0+(n+m)*pi*sigmay**2)/(2*W**2)
            expargy1 =    2*m*pi*(1j*W*y0+n*pi*sigmay**2)/W**2
            expargy2 =    2*n*pi*(1j*W*y0+m*pi*sigmay**2)/W**2
            expargy3 = 2j*(n+m)*pi*y0/W

            argy1 = (W - y0 - 1j*(m-n)*pi*sigmay**2/W)
            argy2 = (    y0 + 1j*(m-n)*pi*sigmay**2/W)
            argy3 = (W - y0 + 1j*(m-n)*pi*sigmay**2/W)
            argy4 = (    y0 - 1j*(m-n)*pi*sigmay**2/W)
            argy5 = (W - y0 - 1j*(m+n)*pi*sigmay**2/W)
            argy6 = (    y0 + 1j*(m+n)*pi*sigmay**2/W)
            argy7 = (W - y0 + 1j*(m+n)*pi*sigmay**2/W)
            argy8 = (    y0 - 1j*(m+n)*pi*sigmay**2/W)

            argy = [argy1, argy2, argy3, argy4, argy5, argy6, argy7, argy8]
            (argy1, argy2, argy3, argy4,
             argy5, argy6, argy7, argy8) = [ complex(a)/(np.sqrt(2.)*sigmay) for a in argy ]

            Iy = 0.25*np.exp(expargy0) * np.sqrt(np.pi/2.) * sigmay * (-2. +
                    np.exp(expargy1) * (erf(argy1) + erf(argy2)) +
                    np.exp(expargy2) * (erf(argy3) + erf(argy4)) -
                    np.exp(expargy3) * (erf(argy5) + erf(argy6)) +
                    erfc(argy7) + erfc(argy8))

            if self.test_integrals:
                Iyalt = (1/4)*np.exp(1)**((-1)*W**(-2)*np.pi*((m**2+n**2)*np.pi*sigmay**2+ \
                        (1j*2)*W*m*y0))*((1/2)*np.pi)**(1/2)*sigmay*(np.exp(1)**((1/2) \
                        *W**(-2)*np.pi*((m+n)**2*np.pi*sigmay**2+(1j*2)*W*(3*m+(-1)*n)*y0) \
                        )*erf(2**(-1/2)*sigmay**(-1)*(W+(1j*(-1))*W**(-1)*(m+(-1)*n) \
                        *np.pi*sigmay**2+(-1)*y0))+np.exp(1)**((1/2)*W**(-2)*(m+n)*np.pi*( \
                        (m+n)*np.pi*sigmay**2+(1j*2)*W*y0))*erf(2**(-1/2)*sigmay**(-1)*(W+ \
                        1j*W**(-1)*(m+(-1)*n)*np.pi*sigmay**2+(-1)*y0))+(-1)*np.exp(1)**(( \
                        1/2)*W**(-2)*np.pi*((m+(-1)*n)**2*np.pi*sigmay**2+(1j*2)*W*(3*m+n) \
                        *y0))*erf(2**(-1/2)*sigmay**(-1)*(W+(1j*(-1))*W**(-1)*(m+n) \
                        *np.pi*sigmay**2+(-1)*y0))+(-1)*np.exp(1)**((1/2)*W**(-2)*(m+(-1) \
                        *n)*np.pi*((m+(-1)*n)*np.pi*sigmay**2+(1j*2)*W*y0))*erf(2**(-1/2) \
                        *sigmay**(-1)*(W+1j*W**(-1)*(m+n)*np.pi*sigmay**2+(-1)*y0))+ \
                        np.exp(1)**((1/2)*W**(-2)*(m+n)*np.pi*((m+n)*np.pi*sigmay**2+( \
                        1j*2)*W*y0))*erf(2**(-1/2)*sigmay**(-1)*((1j*(-1))*W**(-1)*(m+(-1) \
                        *n)*np.pi*sigmay**2+y0))+np.exp(1)**((1/2)*W**(-2)*np.pi*((m+n) \
                        **2*np.pi*sigmay**2+(1j*2)*W*(3*m+(-1)*n)*y0))*erf(2**(-1/2) \
                        *sigmay**(-1)*(1j*W**(-1)*(m+(-1)*n)*np.pi*sigmay**2+y0))+(-1) \
                        *np.exp(1)**((1/2)*W**(-2)*(m+(-1)*n)*np.pi*((m+(-1)*n) \
                        *np.pi*sigmay**2+(1j*2)*W*y0))*erf(2**(-1/2)*sigmay**(-1)*((1j*( \
                        -1))*W**(-1)*(m+n)*np.pi*sigmay**2+y0))+(-1)*np.exp(1)**((1/2) \
                        *W**(-2)*np.pi*((m+(-1)*n)**2*np.pi*sigmay**2+(1j*2)*W*(3*m+n)*y0) \
                        )*erf(2**(-1/2)*sigmay**(-1)*(1j*W**(-1)*(m+n)*np.pi*sigmay**2+y0) \
                        ))

                print "Ix, Ixalt, |Ix-Ixalt|", Ix, Ixalt, abs(Ix-Ixalt)
                print "Iy, Iyalt, |Iy-Iyalt|", Iy, Iyalt, abs(Iy-Iyalt)

        Gamma = 1./(np.pi*W) * kF * kr / np.sqrt(k(n)*k(m)) * (Ix * Iy)

        return Gamma

    def get_matrix(self, x0, y0):
        Gamma = [self.get_matrix_element(n, m, x0=x0, y0=y0) for n in (1, 2)
                                                              for m in (1, 2)]
        return np.asarray(Gamma).reshape(2,-1)



# class Loss(object):
#     """Position dependent loss class.
#
#     Returns a function which places Gaussian dissipation profiles at
#     coordinates {x0,y0}.
#     """
#
#     def __init__(self, k=None, kF=None, kr=None, W=None,
#                  sigmax=0.01, sigmay=0.01, integrate_R2=True,
#                  test_integrals=False):
#         self.sigmax = sigmax
#         self.sigmay = sigmay
#         self.test_integrals = test_integrals
#         self.integrate_R2 = integrate_R2
#
#         if None in (k, kF, kr, W):
#             raise Exception("Error: need wavenumber/width information!")
#         else:
#             self.k = k
#             self.kF = kF
#             self.kr = kr
#             self.W = W
#
#     def Gamma(self, n, m, x0=0, y0=0):
#         """Return the integrals needed for Gamma_tilde."""
#         k = self.k
#         kF = self.kF
#         kr = self.kr
#         W = self.W
#         T0 = 2.*pi/self.kr
#
#         sigmax = self.sigmax
#         sigmay = self.sigmay
#
#         if self.integrate_R2:
#             Ix = np.exp(1)**((1/2)*(k(m)+(-1)*k(n))*((1j*2)*x0+sigmax**2*((-1)*k(m)+k(n))))
#             Iy = (1/4)*np.exp(1)**((-1/2)*W**(-2)*(m+n)*np.pi*((m+n) \
#                 *np.pi*sigmay**2+(1j*2)*W*y0))*((-1)+(-1)*np.exp(1)**((1j*2)*W**( \
#                 -1)*(m+n)*np.pi*y0)+np.exp(1)**(2*W**(-2)*n*np.pi*( \
#                 m*np.pi*sigmay**2+1j*W*y0))+np.exp(1)**(2*W**(-2)*m*np.pi*( \
#                 n*np.pi*sigmay**2+1j*W*y0)))
#         else:
#             # x integration
#             expargx = -0.5*(k(n)-k(m))*(2j*x0 + (k(n)-k(m)) * sigmax**2)
#             argx1 = (T0 - x0 + 1j*(k(n)-k(m))*sigmax**2)
#             argx2 = (   - x0 + 1j*(k(n)-k(m))*sigmax**2)
#
#             # TODO: why do we need a complex() cast here?
#             argx = [argx1, argx2]
#             argx1, argx2 = [ complex(a)/(np.sqrt(2.)*sigmax) for a in argx ]
#
#             Ix = np.exp(expargx) * np.sqrt(np.pi/2.) * sigmax *  (erf(argx1) - erf(argx2))
#
#             if self.test_integrals:
#                 Ixalt = np.exp(1)**((1/2)*(k(m)+(-1)*k(n))*(((-1)*k(m)+k(n))*sigmax**2+( \
#                         1j*2)*x0))*((1/2)*np.pi)**(1/2)*sigmax*((-1)*erf(2**(-1/2) \
#                         *sigmax**(-1)*((1j*(-1))*(k(m)+(-1)*k(n))*sigmax**2+(-1)*x0))+erf( \
#                         2**(-1/2)*sigmax**(-1)*(2*np.pi/kr+(1j*(-1))*(k(m)+(-1)*k(n)) \
#                         *sigmax**2+(-1)*x0)))
#
#             # y integration
#             expargy0 = -(n+m)*pi*(2j*W*y0+(n+m)*pi*sigmay**2)/(2*W**2)
#             expargy1 =    2*m*pi*(1j*W*y0+n*pi*sigmay**2)/W**2
#             expargy2 =    2*n*pi*(1j*W*y0+m*pi*sigmay**2)/W**2
#             expargy3 = 2j*(n+m)*pi*y0/W
#
#             argy1 = (W - y0 - 1j*(m-n)*pi*sigmay**2/W)
#             argy2 = (    y0 + 1j*(m-n)*pi*sigmay**2/W)
#             argy3 = (W - y0 + 1j*(m-n)*pi*sigmay**2/W)
#             argy4 = (    y0 - 1j*(m-n)*pi*sigmay**2/W)
#             argy5 = (W - y0 - 1j*(m+n)*pi*sigmay**2/W)
#             argy6 = (    y0 + 1j*(m+n)*pi*sigmay**2/W)
#             argy7 = (W - y0 + 1j*(m+n)*pi*sigmay**2/W)
#             argy8 = (    y0 - 1j*(m+n)*pi*sigmay**2/W)
#
#             argy = [argy1, argy2, argy3, argy4, argy5, argy6, argy7, argy8]
#             (argy1, argy2, argy3, argy4,
#              argy5, argy6, argy7, argy8) = [ complex(a)/(np.sqrt(2.)*sigmay) for a in argy ]
#
#             Iy = 0.25*np.exp(expargy0) * np.sqrt(np.pi/2.) * sigmay * (-2. +
#                     np.exp(expargy1) * (erf(argy1) + erf(argy2)) +
#                     np.exp(expargy2) * (erf(argy3) + erf(argy4)) -
#                     np.exp(expargy3) * (erf(argy5) + erf(argy6)) +
#                     erfc(argy7) + erfc(argy8))
#
#             if self.test_integrals:
#                 Iyalt = (1/4)*np.exp(1)**((-1)*W**(-2)*np.pi*((m**2+n**2)*np.pi*sigmay**2+ \
#                         (1j*2)*W*m*y0))*((1/2)*np.pi)**(1/2)*sigmay*(np.exp(1)**((1/2) \
#                         *W**(-2)*np.pi*((m+n)**2*np.pi*sigmay**2+(1j*2)*W*(3*m+(-1)*n)*y0) \
#                         )*erf(2**(-1/2)*sigmay**(-1)*(W+(1j*(-1))*W**(-1)*(m+(-1)*n) \
#                         *np.pi*sigmay**2+(-1)*y0))+np.exp(1)**((1/2)*W**(-2)*(m+n)*np.pi*( \
#                         (m+n)*np.pi*sigmay**2+(1j*2)*W*y0))*erf(2**(-1/2)*sigmay**(-1)*(W+ \
#                         1j*W**(-1)*(m+(-1)*n)*np.pi*sigmay**2+(-1)*y0))+(-1)*np.exp(1)**(( \
#                         1/2)*W**(-2)*np.pi*((m+(-1)*n)**2*np.pi*sigmay**2+(1j*2)*W*(3*m+n) \
#                         *y0))*erf(2**(-1/2)*sigmay**(-1)*(W+(1j*(-1))*W**(-1)*(m+n) \
#                         *np.pi*sigmay**2+(-1)*y0))+(-1)*np.exp(1)**((1/2)*W**(-2)*(m+(-1) \
#                         *n)*np.pi*((m+(-1)*n)*np.pi*sigmay**2+(1j*2)*W*y0))*erf(2**(-1/2) \
#                         *sigmay**(-1)*(W+1j*W**(-1)*(m+n)*np.pi*sigmay**2+(-1)*y0))+ \
#                         np.exp(1)**((1/2)*W**(-2)*(m+n)*np.pi*((m+n)*np.pi*sigmay**2+( \
#                         1j*2)*W*y0))*erf(2**(-1/2)*sigmay**(-1)*((1j*(-1))*W**(-1)*(m+(-1) \
#                         *n)*np.pi*sigmay**2+y0))+np.exp(1)**((1/2)*W**(-2)*np.pi*((m+n) \
#                         **2*np.pi*sigmay**2+(1j*2)*W*(3*m+(-1)*n)*y0))*erf(2**(-1/2) \
#                         *sigmay**(-1)*(1j*W**(-1)*(m+(-1)*n)*np.pi*sigmay**2+y0))+(-1) \
#                         *np.exp(1)**((1/2)*W**(-2)*(m+(-1)*n)*np.pi*((m+(-1)*n) \
#                         *np.pi*sigmay**2+(1j*2)*W*y0))*erf(2**(-1/2)*sigmay**(-1)*((1j*( \
#                         -1))*W**(-1)*(m+n)*np.pi*sigmay**2+y0))+(-1)*np.exp(1)**((1/2) \
#                         *W**(-2)*np.pi*((m+(-1)*n)**2*np.pi*sigmay**2+(1j*2)*W*(3*m+n)*y0) \
#                         )*erf(2**(-1/2)*sigmay**(-1)*(1j*W**(-1)*(m+n)*np.pi*sigmay**2+y0) \
#                         ))
#
#                 print "Ix, Ixalt, |Ix-Ixalt|", Ix, Ixalt, abs(Ix-Ixalt)
#                 print "Iy, Iyalt, |Iy-Iyalt|", Iy, Iyalt, abs(Iy-Iyalt)
#
#         Gamma = 1./(2.*np.pi*W) * kF * kr / np.sqrt(k(n)*k(m)) * (Ix * Iy)
#
#         return Gamma
#
#     def get_Gamma_tilde(self, x0, y0):
#         """Return the Gamma_tilde matrix."""
#
#         Gamma_tilde = [ self.Gamma(n, m, x0=x0, y0=y0) for n in (1, 2)
#                                                         for m in (1, 2) ]
#         return np.asarray(Gamma_tilde).reshape(2,-1)


if __name__ == '__main__':
    N = 2.5
    W = 1
    kF = N*np.pi/W
    k = lambda n: np.sqrt(kF**2 - (n*np.pi/W)**2)
    kr = k(1)-k(2)

    L = Loss(k=k, kF=kF, kr=kr, W=W, test_integrals=True)
    print L.Gamma(1,1)
    print L.Gamma(1,2, 0.0923, 0.1231)
    print L.Gamma(2,1, 0.23, 0.31)
    print L.Gamma(2,2, 0.923, 0.1231)
