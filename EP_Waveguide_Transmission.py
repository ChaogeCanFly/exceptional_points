#!/usr/bin/env python2.7

import numpy as np
from numpy import pi

class Transmission_Matrix:
    """Transmission matrix class."""
    
    def __init__(self, N=1.01, eta=0.1, delta=0.0, eps=None, eps_factor=1.0, L=50,
                 infile="S_matrix.dat"):
        
        self.k0, self.k1 = [ np.sqrt(N**2 - n**2)*pi for n in (0,1) ]
        self.kr = self.k0 - self.k1
        
        self.eps = eta/(2*np.sqrt(2*self.k0*self.k1))
        if eps:
            self.eps = eps
        else:
            self.eps *= eps_factor
    
        self.N = N
        self.eta = eta
        self.delta = delta
        self.L = L
        self.x = np.arange(0,L,0.1)
        self.beta = np.sqrt(self.k1/(2*self.k0))
        self.B = self.kr * np.sqrt(self.k0/(2*self.k1))
        
        self.H11 = -self.k0 - 1j*self.eta/2.
        self.H22 = -self.k0 - self.delta - 1j*self.eta/2.*self.k0/self.k1
        self.H12 = self.B * (-1j) * self.eps
        self.H21 = self.B * (+1j) * self.eps
        self.D = np.sqrt((self.H11-self.H22)**2 + 4*self.H12*self.H21)
        self.lam1 = 0.5*(self.H11 + self.H22) + 0.5*self.D
        self.lam2 = 0.5*(self.H11 + self.H22) - 0.5*self.D
        self.gam1 = self.lam1 - self.H11
        self.gam2 = self.lam2 - self.H11
        
        self.infile = infile
        self.S = _extract_S_matrix()
        
    def get_U_at_EP(self):
        """Return the elements of the evolution operator U(x) at the
        exceptional point (EP).
        
            Parameters:
            -----------
                None
                
            Returns:
            --------
                U: (2,2) ndarray
                    Evolution operator.
        """
        exp = np.exp(-1j*x*0.5*(self.H11+self.H22))
        
        M11 = 1. + np.sqrt(self.H12*self.H21)*x
        M12 = -1j*self.H12*x
        M21 = -1j*self.H21*x
        M22 = 1. - np.sqrt(self.H12*self.H21)*x
        
        M11, M12, M21, M22 = [ exp*m for m in (M11,M12,M21,M22) ]
        
        #return M11, M12*self.beta, M21/self.beta, M22
        return M11, M12, M21, M22
    
    def get_U(self):
        """Return the elements of the evolution operator U(x) away from the
        exceptional point (EP).
        
            Parameters:
            -----------
                None
                
            Returns:
            --------
                U: (2,2) ndarray
                    Evolution operator.
        """
        
        pre = np.exp(-1j*self.lam1*x)/(self.D)
        exp = np.exp(1j*self.D*x)
        M11 = self.gam1*exp - self.gam2
        M12 = (1 - exp)*self.H12
        M21 = (1 - exp)*self.H21
        M22 = self.gam1 - self.gam2*exp
        
        M11, M12, M21, M22 = [ pre*m for m in (M11,M12,M21,M22) ]
        
        #return M11, M12*self.beta, M21/self.beta, M22
        return M11, M12, M21, M22
   
    def _extract_S_matrix(self):
        """Extract and return the elements of the scattering matrix S."""
        
        return np.loadtxt(self.infile, unpack=True,
                          dtype=complex, usecols=(1,6,7,8,9))
   

def compare_H_eff_to_S_matrix(**kwargs):
    """Plot the length-dependent S-matrix and compare it to the effective model
    predictions of the evolution operator.
    """
    T = Transmission_Matrix(**kwargs)
    
def parse_arguments():
    """Parse input for Transmission class.
        
        Parameters:
        -----------
            None
            
        Returns:
        --------
            kwargs: dict
    """
    import json
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter as help_formatter
    
    parser = argparse.ArgumentParser(formatter_class=help_formatter)
    
    parser.add_argument("--eta", nargs="?", default=0.0, type=float,
                        help="Dissipation coefficient" )
    parser.add_argument("-L", nargs="?", default=100, type=float,
                        help="Waveguide length" )
    parser.add_argument("--N", nargs="?", default=1.01, type=float,
                        help="Number of open modes int(k*d/pi)" )
    parser.add_argument("--eps-factor", nargs="?", default=1.0, type=float,
                        help="Constant to shift x_EP -> x_EP * eps_factor" )
    parser.add_argument("--eps", nargs="?", default=None, type=float,
                        help="Set value for x_EP to eps (only done if not None)" )
    parser.add_argument("-d", "--delta", nargs="?", default=0.0, type=float,
                        help="Constant to set y_EP (or, equivalently, y_EP -> y_EP + delta)" )
    parser.add_argument("-i", "--infile", default="S_matrix.dat", type=str,
                        help="Input file which contains the S-matrix.")
    
    args = parser.parse_args()
    
    return vars(args)
    

if __name__ == '__main__':
    kwargs = parse_arguments()    
