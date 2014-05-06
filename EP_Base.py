#!/usr/bin/env python

from EP_Helpers import c_eig, c_trapz
from scipy.integrate import complex_ode
import numpy as np
from numpy import pi

class EP_Base:
    """EP_Base class."""
    
    def __init__(self, T=10, x_EP=0.0, y_EP=0.0, x_R0=0.8, y_R0=0.8,
                 loop_type="Circle", loop_direction='-', init_state='a',
                 init_loop_phase=0.0, calc_adiabatic_state=False):
        """Exceptional Point (EP) base class.
        
        The dynamics of a 2-level system are determined via an Runge-Kutta method
        of order (4) 5 due to Dormand and Prince. 
        
            Parameters:
            -----------
                T : float, optional
                    Total duration of the loop in parameter space
                x_EP : float, optional
                    x-corrdinate of loop center
                y_EP : float, optional
                    y-corrdinate of loop center
                x_R0 : float, optional
                    Maximum distance between trajectory and EP in x-direction
                y_R0 : float, optional
                    Maximum distance between trajectory and EP in y-direction
                init_state : str, optional
                    Determines initial state for the system's evolution
                loop_type : str, optional
                    Loop trajectory shape
                loop_direction : str, optional ("-"|"+")
                    Direction of evolution around the EP
                init_loop_phase : float, optional
                    Starting point of evolution on trajectory
                calc_adiabatic_state : bool, optional
                    Wheather adiabatic solutions should also be calculated (note
                    that setting this flag True can slow down the computation
                    considerably)
                
            Returns:
            --------
            None
        
        """
        
        # total loop duration
        self.T = T
        
        # direction of loop parametrization:
        #       +: clockwise,
        #       -: anticlockwise
        self.loop_direction = loop_direction
        
        # loop frequency
        self.w = 2.*pi/T
        if self.loop_direction == '+':
            self.w = -self.w
        
        # number of timesteps in ODE-integration 
        self.tN = T * 5e2 * 1.
         
        # choose between different loop parametrizations
        self.loop_type = loop_type
        
        # determines initial state:
        #       'a': populate gain state |a>
        #       'b': populate loss state |b>
        #       'c': superposition of gain and loss state 2^(-1/2)*(|a> + |b>)
        self.init_state = init_state
        
        # loop cycle parameters
        self.x_EP, self.x_R0 = x_EP, x_R0
        self.y_EP, self.y_R0 = y_EP, y_R0
        self.init_loop_phase = init_loop_phase
        
        # time-array and step-size 
        self.t, self.dt = np.linspace(0, T, self.tN, retstep=True)
        
        # wavefunction |Psi(t)>
        self.Psi = np.zeros((self.tN,2), dtype=np.complex256)
        
        # instantaneous eigenvalues E_a, E_b and 
        # corresponding eigenvectors |phi_a> and |phi_b>
        self.eVals = np.zeros((self.tN,2), dtype=np.complex256)
        self.eVecs_r = np.zeros((self.tN,2,2), dtype=np.complex256)
        self.eVecs_l = np.zeros((self.tN,2,2), dtype=np.complex256)
        
        # adiabatic coefficient and adiabatic phase
        self.Psi_adiabatic = np.zeros((self.tN,2), dtype=np.complex256)
        self.theta = np.zeros((self.tN,2), dtype=np.complex256)
        # flag to switch off the calculation of the adiabatic state
        # this quantity is not needed for heatmap runs and slows down
        # the code considerably
        self.calc_adiabatic_state = calc_adiabatic_state
 
 
    def get_cycle_parameters(self, t):
        """get_cycle_parameters is overwritten by inherited classes."""
        pass


    def H(self, t):
        """H method is overwritten by inherited classes."""
        pass
    
    
    def get_c_eigensystem(self):
        """Calculate the instantaneous eigenvalues and eigenvectors for
        all times t=0,...,T and remove any discontinuities.
        
            Parameters:
            -----------
                None
                
            Returns:
            --------
                None
                
        """
        
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
        epsilon = 1e-3
        mask = abs(diff) > epsilon
        
        # 3) assemble the arrays in a piecewise fashion at points
        #    where eigenvalue-jumps occur
        for k in mask.nonzero()[0]:
            # correct phase to obtain continuous wavefunction
            phase_0_R = np.angle(eVecs_r[k,:,0]) - np.angle(eVecs_r[k+1,:,1])
            phase_0_L = np.angle(eVecs_l[k,:,0]) - np.angle(eVecs_l[k+1,:,1])
            phase_1_R = np.angle(eVecs_r[k+1,:,0]) - np.angle(eVecs_r[k,:,1])
            phase_1_L = np.angle(eVecs_l[k+1,:,0]) - np.angle(eVecs_l[k,:,1])
            #phase_0_R = phase_1_R = 0.0
            #phase_0_L = phase_1_L = 0.0
            #print "phase_0: ", phase_0_R/pi
            #print "phase_1: ", phase_1_R/pi
            
            
            ##### plot phases and moduli of left and right wavefunctions
            ##### before and after the jumps are fixed
            #####clf()
            #####_, ((ax1,ax2),(ax3,ax4)) = subplots(nrows=2, ncols=2)
            #####
            #####ax1.plot(angle(eVecs_l[:,0,0])/pi,"r-")
            #####ax1.plot(angle(eVecs_l[:,1,0])/pi,"g-")
            #####ax2.plot(abs(eVecs_l[:,0,0])/pi,"r-")
            #####ax2.plot(abs(eVecs_l[:,1,0])/pi,"g-")
            #####ax3.plot(angle(eVecs_l[:,0,1])/pi,"r-")
            #####ax3.plot(angle(eVecs_l[:,1,1])/pi,"g-")
            #####ax4.plot(abs(eVecs_l[:,0,1])/pi,"r-")
            #####ax4.plot(abs(eVecs_l[:,1,1])/pi,"g-")
            
            # account for phase-jump v0(k) -> v1(k+1)
            eVecs_r[k+1:,:,1] *= np.exp(+1j*phase_0_R)
            eVecs_l[k+1:,:,1] *= np.exp(+1j*phase_0_L)
            # account for phase-jump v1(k) -> v0(k+1)
            eVecs_r[:k+1,:,1] *= np.exp(+1j*phase_1_R)
            eVecs_l[:k+1,:,1] *= np.exp(+1j*phase_1_L)
            
            for e in eVals, eVecs_r, eVecs_l:
                e[...,0], e[...,1] = (
                                np.concatenate((e[:k+1,...,0], e[k+1:,...,1])),
                                np.concatenate((e[:k+1,...,1], e[k+1:,...,0]))
                                )

            #####ax1.plot(angle(eVecs_l[:,0,0])/pi,"b--")
            #####ax1.plot(angle(eVecs_l[:,1,0])/pi,"k--")
            #####ax2.plot(abs(eVecs_l[:,0,0])/pi,"b--")
            #####ax2.plot(abs(eVecs_l[:,1,0])/pi,"k--")
            #####ax3.plot(angle(eVecs_l[:,0,1])/pi,"b--")
            #####ax3.plot(angle(eVecs_l[:,1,1])/pi,"k--")
            #####ax4.plot(abs(eVecs_l[:,0,1])/pi,"b--")
            #####ax4.plot(abs(eVecs_l[:,1,1])/pi,"k--")
            #####show()
        
        #print np.einsum('ijk,ijk -> ik', eVecs_l, eVecs_r)
        
        self.eVals = eVals
        self.eVecs_l = eVecs_l
        self.eVecs_r = eVecs_r
        
    
    def get_adiabatic_state(self, n):
        """Calculate the adiabatical dynamic phase factor exp(1j*theta).
        
            Parameters:
            -----------
                n: integer
                    Determines the upper integral boundary value t[n] < T.
                    
            Returns:
            --------
                dynamical phase: float
                
        """
        
        E_a, E_b = [ self.eVals[:n,i] for i in (0,1) ]
        
        self.theta[n,:] = [ -c_trapz(E, dx=self.dt) for E in (E_a,E_b) ]
        exp_a, exp_b = [ np.exp(1j*theta) for theta in self.theta[n,:] ]
        
        return exp_a, exp_b
            

    def get_gain_state(self):
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

    
    def get_init_state(self):
        """Return the initial state vector at time t=0.
        
        Depending on the self.init_state variable, a vector |phi_i(0)> is
        returned, with i = a, b or c/d (= linear combinations of a and b).
        
            Parameters:
            -----------
                None
                
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
            #phase = exp(1j*0.77185547)
            phase = np.exp(1j*pi)
            eVec0_r = self.eVecs_r[0,:,0] + phase*self.eVecs_r[0,:,1]
            norm = lambda vl, vr: np.sqrt(vl.dot(vr))
            eVec0_r /= norm(eVec0_r.conj(), eVec0_r)
        
        return eVec0_r
 
   
    def solve_ODE(self):
        """Iteratively solve the ODE dy/dt = f(t,y) on a discretized time-grid.

            Parameters:
            -----------
                    None
                    
            Returns:
            --------
                    t:  (N,)  ndarray
                        Time array.
                phi_a:  (N,2) ndarray
                        Overlap <phi_a|psi>.
                phi_b:  (N,2) ndarray
                        Overlap <phi_b|psi>.
        """
        
        # r.h.s of ode d/dt y = f(t, y)
        f = lambda t, phi: -1j*self.H(t).dot(phi)
        
        # create ode object to solve Schroedinger equation (SE)
        SE = complex_ode(f).set_integrator('dopri5', rtol=1e-9)
        
        # set initial conditions
        self.get_c_eigensystem()    # calculate eigensystem for all times
        self.get_gain_state()       # find state with total (relative) gain
        self.eVec0 = self.get_init_state()          # define initial state
        SE.set_initial_value(self.eVec0, t=0.0)     # y0, t0
                
        # iterate ode
        for n, tn in enumerate(self.t):
            if SE.successful():
                self.Psi[n,:]  = SE.y
                if self.calc_adiabatic_state:
                    self.Psi_adiabatic[n,:] = self.get_adiabatic_state(n)
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
