#!/usr/bin/env python

from ep.base import Base
import numpy as np
from numpy import pi

class Toymodel(Base):
    """EP_Toymodel class."""
    
    def __init__(self, **kwargs):
        """Copy methods and variables from EP_Base class."""
        EP_Base.__init__(self, **kwargs)


    def H(self, t):
        """Return parametrically dependent Hamiltonian at time t.
        
            Parameters:
            -----------
                t: float
                
            Returns:
            --------
                H: (2,2) ndarray
                
        """
        
        # Pauli matrices sigma_x and sigma_z
        sigma_x = np.array([[0,1], [1,0]], dtype=complex)
        sigma_z = np.array([[1,0], [0,-1]], dtype=complex)
        # unperturbed Hamiltonian H_0
        H_0 = np.array([[-1,1j], [1j,1]], dtype=complex)
        
        c1, c2 = self.get_cycle_parameters(t)
        
        return H_0 + c1*sigma_x + c2*sigma_z


    def get_cycle_parameters(self, t):
        """Return the loop parameters at time t.
        
            Parameters:
            -----------
                t: float
                    Time t.
                    
            Returns:
            --------
                x, y: float
                    Trajectory coordinates (x,y) at time t.
        
        """
        
        a1, b1 = self.x_EP, self.x_R0
        a2, b2 = self.y_EP, self.y_R0
        c1, c2 = 0, 0
        p0 = self.init_loop_phase

        def pacman_shape(t):
            p1, p2 = 1*pi/2 - 1.0, 2.0
            t1, t2 = [ p/self.w for p in p1, p2 ]

            phi = self.w*t + p0
            phase = np.mod(phi, 2*pi)
            R = lambda phi: 0.5 / (p2/2.)**2 * (phi - (p1 + p2/2.))**2 + 0.5
            
            if phase > p1 and phase < (p1 + p2):
                return R(phase)
            else:
                return 1.0
            
        pacman_shape = np.vectorize(pacman_shape)
        
        if self.loop_type is "Circle":
            R = lambda t: 1.
            c1 = c2 = 0.0
        elif self.loop_type is "Circle_wedge":
            R = pacman_shape
            c1 = c2 = 0.0
        elif self.loop_type is "Circle_whirl":
            R = lambda t: 1.
            R = lambda t: (1. + 0.5*np.cos(5*self.w*t))
            c1 = c2 = 0.2
            c1 = c2 = 0.0
        elif self.loop_type is "Dropshape":
            R = lambda t:  (4./self.T**2 * (t - self.T/2)**2 + 2./4.) 
            c1 = c2 = 0.0
        elif self.loop_type is "Dropshape_whirl":
            R = lambda t:  (4./self.T**2 * (t - self.T/2)**2 + 1./4.)
        else:
            raise Exception("""Error: loop_type %s
                                does not exist!""" % self.loop_type)
        
        lambda1 = lambda t: a1 + b1*R(t)*np.cos(self.w*t + p0) + c1*np.cos(10*self.w*t)
        lambda2 = lambda t: a2 + b2*R(t)*np.sin(self.w*t + p0) + c2*np.sin(10*self.w*t)

        return lambda1(t), lambda2(t)
    
    
    def plot_data_thief(self, init_state=True):
        """Plot results from numerical integration and compare to corresponding
        data of Fig. 2 from paper.
        
            Parameters:
            -----------
                init_state: bool
                    
            Returns:
            --------
                None
                
        """
        
        xlabel("time [a.u.]")
        ylabel("Amplutide")
        
        dtf_a1_x, dtf_a1_y = np.loadtxt("datathief/datathief_fig_a_a1t.dat",
                                        unpack=True)
        dtf_b1_x, dtf_b1_y = np.loadtxt("datathief/datathief_fig_a_b1t.dat",
                                        unpack=True, delimiter=",")
        dtf_a2_x, dtf_a2_y = np.loadtxt("datathief/datathief_fig_b_a2t.dat",
                                        unpack=True, delimiter=",")
        dtf_b2_x, dtf_b2_y = np.loadtxt("datathief/datathief_fig_b_b2t.dat",
                                        unpack=True, delimiter=",")
        
        if not init_state:
            title(r"$\varphi_0$ = gain state")
            semilogy(dtf_a1_x, dtf_a1_y, "k-o",  label=r"$a_1(t)$")
            semilogy(dtf_b1_x, dtf_b1_y, "k-s", label=r"$b_1(t)$")
        else:
            title(r"$\varphi_0$ = loss state")
            semilogy(dtf_a2_x, dtf_a2_y, "k-o",  label=r"$a_2(t)$")
            semilogy(dtf_b2_x, dtf_b2_y, "k-s", label=r"$b_2(t)$")
            

if __name__ == '__main__':
    pass
