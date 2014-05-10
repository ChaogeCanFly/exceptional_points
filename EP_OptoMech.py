#!/usr/bin/env python

from EP_Base import *
import numpy as np
from numpy import pi

class EP_OptoMech(EP_Base):
    """EP_OptoMech class."""
    
    def __init__(self, R=0.05, gamma=2.0, **kwargs):
        """
        Exceptional Points (EP) optomechanics class. Copies methods and
        variables from EP_Base class.
        
            Additional parameters:
            ----------------------
                R: float
                    Trajectory radius around the EP.
                gamma: float
                    Relative loss between states |1> and |2>.
                    
        """
        EP_Base.__init__(self, **kwargs)
        self.R = R
        self.gamma = gamma
        self.x_EP = 0.0
        self.y_EP = gamma/2.
       
       
    def H(self, t, x=None, y=None):
        """
        Return parametrically dependent Hamiltonian at time t. The exact form
        of H has been taken from eq. (1) in the paper draft of Thomas J. Milburn
        (2014-04-18).
        
            Parameters:
            -----------
                t: float
                
            Returns:
            --------
                H: (2,2) ndarray
                
        """
        if x is None and y is None:
            omega, g = self.get_cycle_parameters(t)
        else:
            omega, g = x, y
        
        H11 = - omega - 1j * self.gamma/2.
        H12 = g
        H21 = H12
        H22 = - H11
        
        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H


    def get_cycle_parameters(self, t):
        """
        Return path around the EP at (omega, g) = (0, gamma/2) parametrized
        via time t.
        
            Parameters:
            -----------
                t, float
                
            Returns:
            --------
                omega: float
                g: float
        """
        
        phi = self.init_loop_phase + self.w*t
        
        omega = self.R * np.sin(phi)
        g = self.R * np.cos(phi) + self.gamma/2.
        
        return omega, g
    
    
    def sample_H(self):
        """Sample local eigenvalue geometry of H.
        
            Returns:
            --------
                X, Y: (N,N) ndarray
                Z: (N,N,2) ndarray
                
        """
        
        xN = yN = 1*10**2
        xEP = self.x_EP
        yEP = self.y_EP
        print xEP, yEP

        x = np.linspace(xEP - 2.2*self.R,
                        xEP + 2.2*self.R, xN)
        y = np.linspace(yEP - 2.2*self.R,
                        yEP + 2.2*self.R, yN)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((xN,yN,2), complex)
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                Z[i,j,:] = c_eig(self.H(0,xi,yj))[0]
                
        # circumvent indexing='ij' option in np.meshgrid
        return X.T, Y.T, Z
    
    
    def get_non_adiabatic_coupling(self):
        """
        Return the non-adiabatic coupling defined as
        
            <1(t)|dH/dt|2(t)> = <2(t)|dH/dt|1(t)>,
            
        where |1(t)> and |2(t)> are the instantaneous eigenstates at time t.
        
            Parameters:
            -----------
                None
                
            Returns:
            --------
                f: (N,) ndarray
                    Non-adiabatic coupling parameter as a function of time t.
                    
        """
        
        e = self.eVals[:,0]
        delta, kappa = self.get_cycle_parameters(self.t)
        D = delta + 1j*kappa
        G = self.G * np.ones_like(D)
        ep, Dp, Gp = [ c_gradient(x,self.dt) for x in (e,D,G) ]        

        f = ((ep - Dp) * G - (e - D) * Gp)/(2.*e*(e - D))

        return f
        
def plot_non_adiabatic_coupling():
    OM = EP_OptoMech(T=64., R=1./16., init_state='b')
    OM.w = 2.*pi/(OM.T/3.)
    t, b0, b1 = OM.solve_ODE()
    b0_ad, b1_ad = (OM.Psi_adiabatic[:,0],
                    OM.Psi_adiabatic[:,1])
    print t[-1], 2*pi/OM.w
    #semilogy(t, abs(b0), "r-")
    #semilogy(t, abs(b1), "g-")
    f = OM.get_non_adiabatic_coupling()
    e = OM.eVals[:,0]
    
    ylim(-3,10)
    #plot(OM.t, -real(b1/b0*f)/imag(e), "r-")
    #plot(OM.t, -real(b0/b1*f)/imag(e), "g-")
    plot(OM.t, -real(f*b0/b1)/imag(e), "r-")
    plot(OM.t, -real(f*b1/b0)/imag(e), "g-")
    #plot(OM.t, abs(b0/b1), "g-")
    show()
    


if __name__ == '__main__':
    evolutions = 5
    OM = EP_OptoMech(T=100.*evolutions, R=1/20., gamma=2., init_state='b',
                     init_loop_phase=pi, loop_direction='+')
    OM.w *= evolutions
    t, cp, cm = OM.solve_ODE()
    R = abs(cp/cm)
    plot(t,R**(-1),"r-")
    #semilogy(t, abs(cp), "r-")
    #semilogy(t, abs(cm), "g-")
    show()
