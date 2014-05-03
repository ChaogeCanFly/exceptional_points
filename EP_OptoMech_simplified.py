#!/usr/bin/env python

from EP_Base import *

class EP_OptoMech_simplified(EP_Base):

    def __init__(self, R=0.25, **kwargs):
        EP_Base.__init__(self, **kwargs)
        self.R = R

    def H(self, t):
        H11 = 1j*(1 - self.R*np.exp(1j*self.w*t))
        H12 = 1
        H21 = 1
        H22 = -H11
        
        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H

    def get_init_state(self):
        #return np.array([1,0])
        return np.array([0,1])


if __name__ == '__main__':
    OM = EP_OptoMech_simplified(T=32, R=1./4.)
    t, _, _ = OM.solve_ODE()
    # original basis coefficients
    b1, b2 = OM.Psi[:,0], OM.Psi[:,1]
    # eigenvalues
    eval1, eval2 = OM.eVals[:,0], OM.eVals[:,1]
    # eigenvectors
    evec11, evec12 = OM.eVecs_r[:,0,0], OM.eVecs_r[:,1,0]
    evec21, evec22 = OM.eVecs_r[:,0,1], OM.eVecs_r[:,1,1]
    np.savetxt("OptoMech_simplified_H_b1_0_b2_1.dat", zip(t, b1, b2, eval1, eval2,
                                                          evec11, evec12, evec21,
                                                          evec22))
    #semilogy(t, abs(b1), "r-",
    #         t, abs(b2), "g-")
    #show()
    
