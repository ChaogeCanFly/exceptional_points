#!/usr/bin/env python

from EP_Base import *

class EP_OptoMech(EP_Base):
    """
    """
    def __init__(self, R=0.25, chi=5.*pi/4., q=0., **kwargs):
        """Copy methods and variables from EP_Base class.
        
            Additional parameters:
                R: float
                    Trajectory radius around the EP.
        """
        EP_Base.__init__(self, **kwargs)
        self.R = R
        self.chi = chi
        self.q = q

        self.B1 = -np.exp(1j*self.chi) / (-1j*2 + self.q) 
        self.B2 = -np.exp(1j*self.chi) / (+1j*2 + self.q)
        

    def H(self, t, x=None, y=None):
        """
        Return parametrically dependent Hamiltonian at time t. The exact form
        of H has been taken from eq. (54) in the notes of Thomas J. Milburn
        (2013-11-21).
        
            Parameters:
                t: float
            Returns:
                H: (2,2) ndarray
        """
        if x is None and y is None:
            B_real, B_imag = self.get_cycle_parameters(t)
            B = B_real + 1j*B_imag
        else:
            B = x + 1j*y
        
        H11 = np.exp(1j*self.chi) + self.q*B
        H12 = 2.*B
        H21 = H12
        H22 = -H11
        
        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H
       

    def sample_H(self):
        """
        Sample local eigenvalue geometry of H.
        
            Returns:
                X, Y: (N,N) ndarray
                Z: (N,N,2) ndarray
        """
        xN = yN = 5*10**2
        xEP = np.real(self.B1)
        yEP = np.imag(self.B1)

        x = np.linspace(xEP - 1.2*self.R,
                        xEP + 1.2*self.R, xN)
        y = np.linspace(yEP - 1.2*self.R,
                        yEP + 1.2*self.R, yN)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((xN,yN,2), complex)
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                Z[i,j,:] = c_eig(self.H(0,xi,yj))[0]
                
        # circumvent indexing='ij' option in np.meshgrid
        return X.T, Y.T, Z
    
    
    def get_cycle_parameters(self, t):
        """
        """
        
        phi = lambda t: self.init_loop_phase + self.w*t
        
        B = self.B1 + self.R * np.exp(1j*phi(t))
        
        return np.real(B), np.imag(B)


if __name__ == '__main__':
    pass
