#!/usr/bin/env python2.7

from __future__ import division
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
try:
    from mayavi import mlab
except:
    print "Warning: mayavi not found!"
from scipy.integrate import complex_ode

from ep.helpers import c_eig, c_trapz, map_trajectory


class Base:
    """Base class."""

    def __init__(self, T=100, tN=None, x_R0=0.05, y_R0=0.4, loop_type="Circle",
                 loop_direction='-', init_state='a', init_phase=0.0,
                 calc_adiabatic_state=False, verbose=False):
        """Exceptional Point (EP) base class.

        The dynamics of a 2-level system are determined via an Runge-Kutta
        method of order (4) 5 due to Dormand and Prince.

            Parameters:
            -----------
                T : float, optional
                    Total duration of the loop in parameter space.
                tN: int, optional
                    Number if timesteps in the ODE-integration.
                x_R0 : float, optional
                    x-coordinate of the loop parametrization.
                y_R0 : float, optional
                    y-coordinate of the loop parametrization.
                init_state : str, optional
                    Determines initial state for the system's evolution:
                       'a': populate gain state |a>
                       'b': populate loss state |b>
                       'c': superposition of gain and loss state 2^(-1/2)*(|a> + |b>)
                       'd': superposition of gain and loss state 2^(-1/2)*(|a> - |b>)
                loop_type : str, optional
                    Loop trajectory shape.
                loop_direction : str, optional ("-"|"+")
                    Direction of evolution around the EP (-: counterclockwise, +: clockwise).
                init_phase : float, optional
                    Starting point of evolution on trajectory.
                calc_adiabatic_state : bool, optional
                    Whether adiabatic solutions should also be calculated (note
                    that setting this flag True can slow down the computation
                    considerably).
                verbose: bool, optional
                    Whether to return additional output.
        """
        self.T = T

        self.init_state = init_state
        self.loop_type = loop_type
        self.loop_direction = loop_direction

        # number of timesteps in ODE-integration
        if tN is None:
            tN = 5e1 * T
        self.tN = tN

        # time-array and step-size
        self.t, self.dt = np.linspace(0, T, self.tN, retstep=True)

        # loop frequency
        self.w = 2.*pi/T
        if self.loop_direction == '+':
            self.w = -self.w

        # loop cycle parameters
        self.x_R0, self.y_R0 = x_R0, y_R0
        self.init_phase = init_phase

        # wavefunction |Psi(t)>
        self.Psi = np.zeros((self.tN,2), dtype=np.complex256)

        # instantaneous eigenvalues E_a, E_b and corresponding eigenvectors
        # |phi_a> and |phi_b>
        self.eVals = np.zeros((self.tN,2), dtype=np.complex256)
        self.eVecs_r = np.zeros((self.tN,2,2), dtype=np.complex256)
        self.eVecs_l = np.zeros((self.tN,2,2), dtype=np.complex256)

        # adiabatic coefficient and adiabatic phase
        self.Psi_adiabatic = np.zeros((self.tN,2), dtype=np.complex256)
        self.theta = np.zeros((self.tN,2), dtype=np.complex256)

        self.calc_adiabatic_state = calc_adiabatic_state
        self.verbose = verbose

    def get_cycle_parameters(self, t):
        """get_cycle_parameters method is overwritten by inheriting classes."""
        pass

    def H(self, t, x=None, y=None):
        """Hamiltonian H is overwritten by inheriting classes."""
        pass

    def sample_H(self, xmin=None, xmax=None, xN=None, ymin=None, ymax=None,
                  yN=None, verbose=False):
        """Sample local eigenvalue geometry of Hamiltonian H.

            Parameters:
            -----------
                xmin, xmax: float
                    Dimensions in x-direction.
                ymin, ymax: float
                    Dimensions in y-direction.
                xN, yN: int
                    Number of sampling points in x and y direction.
                verbose: bool
                    Show additional output.

            Returns:
            --------
                X, Y: (N,N) ndarray
                    Spatial (mesh)grids.
                Z: (N,N,2) ndarray
                    Eigenvalues evaluated on the X/Y grid.
        """

        if xN is None:
            xN = 5*10**2
        if yN is None:
            yN = 5*10**2

        # if xmin is None or xmax is None:
        #     xmin = self.x_EP - 0.15*self.x_R0
        #     xmax = self.x_EP + 0.15*self.x_R0
        #
        # if ymin is None or ymax is None:
        #     ymin = self.y_EP - 0.15*self.y_R0
        #     ymax = self.y_EP + 0.15*self.y_R0

        x = np.linspace(xmin, xmax, xN)
        y = np.linspace(ymin, ymax, yN)

        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.zeros((xN,yN,2), dtype=complex)

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                if verbose:
                    print "(i,j) =", i, j
                H = self.H(0,xi,yj)
                Z[i,j,:] = c_eig(H)[0]

        return X, Y, Z

    def plot_3D_spectrum(self, xmin=None, xmax=None, xN=None, ymin=None,
                         ymax=None, yN=None, trajectory=False, part='imag'):
        """Plot the Riemann sheet structure around the EP.

            Parameters:
            -----------
                xmin, xmax: float
                    Dimensions in x-direction.
                ymin, ymax: float
                    Dimensions in y-direction.
                xN, yN: int
                    Number of sampling points in x and y direction.
                trajectory: bool
                    Whether to include a projected trajectory of the eigenbasis
                    coefficients.
                part: str
                    Which function to apply to the eigenvalues before plotting.
        """
        X, Y, Z = self.sample_H(xmin, xmax, xN, ymin, ymax, yN)

        Z0, Z1 = [ Z[...,n] for n in 0, 1 ]

        mlab.figure(0)
        mlab.surf(X.real, Y.real, Z0.real)
        mlab.surf(X.real, Y.real, Z1.real)
        mlab.axes(zlabel="Re(E)")

        mlab.figure(1)
        mlab.mesh(X.real, Y.real, Z0.imag)
        mlab.mesh(X.real, Y.real, Z1.imag)
        mlab.axes(zlabel="Im(E)")

        mlab.figure(2)
        mlab.mesh(X.real, Y.real, abs(Z0-Z1))
        mlab.axes(zlabel="abs(E0-E1)")

        if part == 'imag':
            part = np.imag
        elif part == 'real':
            part = np.real
        else:
            part = np.abs

        if trajectory:
            x, y = self.get_cycle_parameters(self.t)
            _, c1, c2 = self.solve_ODE()
            e1, e2 = [ part(self.eVals[:,n]) for n in 0, 1 ]
            z = map_trajectory(c1, c2, e1, e2)

            mlab.plot3d(x, y, z, tube_radius=5e-6)

        mlab.show()

    def iso_sample_H(self, part=np.real, xmin=None, xmax=None, xN=None,
                     ymin=None, ymax=None, yN=None, zN=None):
        """Sample local eigenvalue geometry of H implicitly.

            Parameters:
            -----------
                xN, yN, zN: int
                    Number of sampling points in x, y and z direction.

            Returns:
            --------
                X, Y, Z, F: (N,N,N) ndarray
        """
        if xN is None:
            xN = 5*10**2
        if yN is None:
            yN = xN
        if zN is None:
            zN = xN

        x = np.linspace(xmin, xmax, xN)
        y = np.linspace(ymin, ymax, yN)

        # x = np.linspace(self.x_EP - 1.1*self.x_R0,
        #                 self.x_EP + 1.1*self.x_R0, xN)
        # y = np.linspace(self.y_EP - 1.1*self.y_R0,
        #                 self.y_EP + 1.1*self.y_R0, yN)

        z = np.linspace(-1,1,zN)

        if part is np.real:
            print "real"
            f = lambda x, E: 1j*np.sign(x)*np.imag(E)
        else:
            print "imag"
            z = 1j*z
            f = lambda x, E: np.sign(x)*np.real(E)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        X, Y = [ np.real(N) for N in X, Y ]

        E = np.zeros((xN,yN,2), dtype=complex)
        F = np.zeros((xN,yN,zN), dtype=complex)

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                H = self.H(0,xi,yj)
                E[i,j,:] = c_eig(H)[0]

                char_poly = np.poly(H)

                for k, zk in enumerate(z):
                    F[i,j,k] = np.polyval(char_poly,
                                          zk + 1j*np.sign(zk)*np.imag(E[i,j,0]))
        return X, Y, Z, F

    def get_c_eigensystem(self):
        """Calculate the instantaneous eigenvalues and eigenvectors for
        all times t=0,...,T and remove any discontinuities."""

        # allocate temporary vectors
        eVals = np.zeros_like(self.eVals)
        eVecs_r = np.zeros_like(self.eVecs_r)
        eVecs_l = np.zeros_like(self.eVecs_l)

        # get eigenvalues and (left and right) eigenvectors at t=tn
        for n, tn in enumerate(self.t):
            eVals[n,:], eVecs_l[n,:,:], eVecs_r[n,:,:] = c_eig(self.H(tn),
                                                               left=True)

        # check for discontinuities of first eigenvalue
        # and switch eigenvalues/eigenvectors accordingly:

        # 1) get differences between array components
        diff = np.diff(eVals[:,0])

        # 2) if difference exceeds epsilon, switch
        epsilon = 1e-1
        mask = abs(diff) > epsilon

        # 3) assemble the arrays in a piecewise fashion at points
        #    where eigenvalue-jumps occur
        for k in mask.nonzero()[0]:
            # correct phase to obtain continuous wavefunction
            phase_0_R = np.angle(eVecs_r[k,:,0]) - np.angle(eVecs_r[k+1,:,1])
            phase_0_L = np.angle(eVecs_l[k,:,0]) - np.angle(eVecs_l[k+1,:,1])
            phase_1_R = np.angle(eVecs_r[k+1,:,0]) - np.angle(eVecs_r[k,:,1])
            phase_1_L = np.angle(eVecs_l[k+1,:,0]) - np.angle(eVecs_l[k,:,1])

            # account for phase-jump v0(k) -> v1(k+1)
            eVecs_r[k+1:,:,1] *= np.exp(+1j*phase_0_R)
            eVecs_l[k+1:,:,1] *= np.exp(+1j*phase_0_L)
            # account for phase-jump v1(k) -> v0(k+1)
            eVecs_r[:k+1,:,1] *= np.exp(+1j*phase_1_R)
            eVecs_l[:k+1,:,1] *= np.exp(+1j*phase_1_L)

            for e in eVals, eVecs_r, eVecs_l:
                e[...,0], e[...,1] = (np.concatenate((e[:k+1,...,0],
                                                      e[k+1:,...,1])),
                                      np.concatenate((e[:k+1,...,1],
                                                      e[k+1:,...,0])))

        #print np.einsum('ijk,ijk -> ik', eVecs_l, eVecs_r)

        self.eVals = eVals
        self.eVecs_l = eVecs_l
        self.eVecs_r = eVecs_r

    def _get_adiabatic_state(self, n):
        """Calculate the adiabatic prediction exp(1j*theta).

            Parameters:
            -----------
                n: integer
                    Determines the upper integral boundary value t[n] < T.

            Returns:
            --------
                adiabatic prediction: float
        """

        E_a, E_b = [ self.eVals[:n,i] for i in (0,1) ]

        self.theta[n,:] = [ -c_trapz(E, dx=self.dt) for E in (E_a,E_b) ]
        exp_a, exp_b = [ np.exp(1j*theta) for theta in self.theta[n,:] ]

        return exp_a, exp_b

    def _get_gain_state(self):
        """Determine the (relative) gain and loss states.

        The integral int_0,T E_a(t) dt is calculated. If the imaginary part of
        the resulting integral is larger than int_0,T E_b(t), E_a is the gain
        state and nothing is done. If not, eigenvalues and eigenstates are
        interchanged.
        """

        # calculate time-integral of both eigenvalues
        intE0, intE1  = [ c_trapz(self.eVals[:,n],
                                  dx=self.dt) for n in (0,1) ]

        # change order of energy eigenvalues and eigenvectors if
        # imag(integral_E0) is smaller than imag(integral_E1)
        if np.imag(intE0) < np.imag(intE1):
            self.eVals[:,:] = self.eVals[:,::-1]
            self.eVecs_r[:,:,:] = self.eVecs_r[:,:,::-1]
            self.eVecs_l[:,:,:] = self.eVecs_l[:,:,::-1]

    def _get_init_state(self):
        """Return the initial state vector at time t=0.

        Depending on the self.init_state variable, a vector |phi_i(0)> is
        returned, with i = a, b or c/d (= linear combinations of a and b).

            Returns:
            --------
                eVec0_r: (2,) ndarray
        """

        if self.init_state == 'a':
            eVec0_r = self.eVecs_r[0,:,0]

        elif self.init_state == 'b':
            eVec0_r = self.eVecs_r[0,:,1]

        elif self.init_state == 'c':
            eVec0_r = self.eVecs_r[0,:,0] + self.eVecs_r[0,:,1]
            eVec0_l = self.eVecs_l[0,:,0] + self.eVecs_l[0,:,1]
            norm = lambda vl, vr: np.sqrt(vl.dot(vr))
            print norm(eVec0_l, eVec0_r)
            print norm(eVec0_r.conj(), eVec0_r)
            eVec0_r /= norm(eVec0_r.conj(), eVec0_r)

        elif self.init_state == 'd':
            phase = np.exp(1j*pi)
            eVec0_r = self.eVecs_r[0,:,0] + phase*self.eVecs_r[0,:,1]
            norm = lambda vl, vr: np.sqrt(vl.dot(vr))
            eVec0_r /= norm(eVec0_r.conj(), eVec0_r)

        return eVec0_r

    def solve_ODE(self, H=None):
        """Iteratively solve the ODE dy/dt = f(t,y) on a discretized time-grid.

            Returns:
            --------
                    t:  (N,)  ndarray
                        Time array.
                phi_a:  (N,2) ndarray
                        Overlap <phi_a|psi>.
                phi_b:  (N,2) ndarray
                        Overlap <phi_b|psi>.
        """

        if H is None:
            H = self.H

        # r.h.s of ode d/dt y = f(t, y)
        f = lambda t, phi: -1j*H(t).dot(phi)

        # create ode object to solve Schroedinger equation (SE)
        SE = complex_ode(f).set_integrator('dopri5', rtol=1e-9, atol=1e-9)

        # set initial conditions
        self.get_c_eigensystem()    # calculate eigensystem for all times
        self._get_gain_state()       # find state with total (relative) gain
        self.eVec0 = self._get_init_state()         # define initial state
        SE.set_initial_value(self.eVec0, t=0.0)     # y0, t0

        # iterate ode
        for n, tn in enumerate(self.t):
            if SE.successful():
                self.Psi[n,:] = SE.y
                if self.calc_adiabatic_state:
                    self.Psi_adiabatic[n,:] = self._get_adiabatic_state(n)
                SE.integrate(SE.t + self.dt)
            else:
                raise Exception("ODE convergence error!")

        # replace projection of states by dot product via Einstein sum
        projection = np.einsum('ijk,ij -> ik',
                               self.eVecs_l, self.Psi)

        self.phi_a, self.phi_b = [ projection[:,n] for n in (0,1) ]

        return self.t, self.phi_a, self.phi_b


if __name__ == '__main__':
    pass
