#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import numpy as np

from ep.base import Base
from ep.helpers import c_gradient


class OptoMech(Base):
    """OptoMech class."""

    def __init__(self, R=0.05, gamma=2.0, **kwargs):
        """Exceptional Points (EP) optomechanics class.

        Copies methods and variables from Base class.

            Additional parameters:
            ----------------------
                R: float
                    Trajectory radius around the EP.
                gamma: float
                    Relative loss between states |1> and |2>.
        """

        Base.__init__(self, **kwargs)
        self.R = R
        self.gamma = gamma
        self.x_EP = 0.0
        self.y_EP = gamma/2.

    def H(self, t, x=None, y=None):
        """Return parametrically dependent Hamiltonian at time t.

        The exact form of H has been taken from eq. (1) in the paper draft of
        Thomas J. Milburn (2014-04-18).

            Parameters:
            -----------
                t: float
                    Time variable.
                x, y: float
                    Parameters in omega-g space.

            Returns:
            --------
                H: (2,2) ndarray
        """

        if x is None and y is None:
            omega, g = self.get_cycle_parameters(t)
        else:
            omega, g = x, y

        H11 = -omega - 1j*self.gamma/2.
        H12 = g
        H21 = H12
        H22 = -H11

        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H

    def get_cycle_parameters(self, t):
        """Return path around the EP at (omega, g) = (0, gamma/2) parametrized
        via time t.

            Parameters:
            -----------
                t, float

            Returns:
            --------
                omega: float
                g: float
        """

        if self.loop_type == "Outside_circle":
            phi = self.init_phase + self.w*t
            omega = self.R * np.sin(phi)
            g = self.R * np.cos(phi) + self.gamma/2. + self.R*0.9
        else:
            phi = self.init_phase + self.w*t
            omega = self.R * np.sin(phi)
            g = self.R * np.cos(phi) + self.gamma/2.

        return omega, g

    def get_non_adiabatic_coupling(self):
        """Return the non-adiabatic coupling defined as

            <1(t)|dH/dt|2(t)> = <2(t)|dH/dt|1(t)>,

        where |1(t)> and |2(t)> are the instantaneous eigenstates at time t.

            Returns:
            --------
                f: (N,) ndarray
                    Non-adiabatic coupling parameter as a function of time t.
        """

        e = self.eVals[:,0]
        delta, kappa = self.get_cycle_parameters(self.t)
        D = delta + 1j*kappa
        G = self.G * np.ones_like(D)
        ep, Dp, Gp = [ c_gradient(x, self.dt) for x in (e, D, G) ]

        f = ((ep - Dp) * G - (e - D) * Gp)/(2.*e*(e - D))

        return f


def plot_figures(fignum='2a', direction='-', show=False,
                 T=45., R=0.1, gamma=1.):

    import brewer2mpl
    cmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
    colors = cmap.mpl_colors

    params = { "T": T, 
               "R": R, 
               "gamma": gamma,
               "loop_type": "Outside_circle"}

    if fignum == '2a':
        settings = { "init_state": 'a',
                     "init_phase": 0,
                     "loop_direction": '+'}
    elif fignum == '2b':
        settings = { "init_state": 'b',
                     "init_phase": 0, 
                     "loop_direction": '-'}
    elif fignum == '2c':
        settings = { "init_state": 'b',
                     "init_phase": np.pi,
                     "loop_direction": '-'}

    params.update(settings)
    OM = OptoMech(**params)

    t, cp, cm = OM.solve_ODE()
    Psi = OM.Psi

    # f, (ax1, ax2) = plt.subplots(ncols=2)
    f, ax1 = plt.subplots()

    ax1.semilogy(t, np.abs(Psi[:,0])**2, ls="-", ms="o", color=colors[0], label=r"$|\alpha_+(t)|^2$")
    ax1.semilogy(t, np.abs(Psi[:,1])**2, ls="-", ms="o", color=colors[1], label=r"$|\alpha_-(t)|^2$")
    ax1.semilogy(t, np.abs(cp)**2, ls="--", color=colors[2], label=r"$|c_+(t)|^2$")
    ax1.semilogy(t, np.abs(cm)**2, ls="--", color=colors[3], label=r"$|c_-(t)|^2$")
    ax1.legend(loc="lower right")
    ax1.set_xlabel(r"$t$")
    m = [ (abs(x)**2).max() for x in Psi, cp, cm ]
    ax1.set_ylim(1e-3, max(m))

    omega, g = OM.get_cycle_parameters(t)
    # np.savetxt("parameters_{}.dat".format(fignum), zip(g, omega))

    ax2 = plt.axes([0.2, 0.65, .2, .2])
    ax2.plot(gamma/2, 0, "ko")
    ax2.plot(g, omega, ls="-", color=colors[0])
    ax2.set_xlim(gamma/4, 3/4.*gamma)
    ax2.set_ylim(-gamma/4., gamma/4.)

    if show:
        plt.show()
    else:
        plt.savefig("{}.png".format(fignum))


if __name__ == '__main__':
    print "Warning: is normalization symmetric?"

    import argh
    argh.dispatch_command(plot_figures)
