#!/usr/bin/env python

from EP_Base import *

class EP_Waveguide_EP_Trajectory(EP_Base):
    """
    """
    def __init__(self, L=100, d=1.0, eta=0.05, N=1.01, theta=0.0, **kwargs):
        """
        Copy methods and variables from EP_Base class and add new variables.
        
            Additional parameters:
                d: float
                    Waveguide width
                eta: float
                    Dissipation coefficient
                N: float
                    Number of open modes
                theta: float
                    Phase difference between upper and lower boundary
        """
        EP_Base.__init__(self, T=L, **kwargs)
        
        self.d = d                                  # wire width
        self.L = L                                  # wire length
        self.eta = eta                              # dissipation coefficient
        #self.position_dependent_eta = False        # use pos. dep. loss
        self.N = N                                  # number of open modes
        self.k0, self.k1 = self.k(0.), self.k(1.)   # longitudinal wavenumbers
                                                    # for mode n=0 and n=1
        self.kr = self.k0 - self.k1                 # wavenumber difference
        
        self.theta_boundary = theta                 # phase angle between upper
                                                    # and lower boundary
        self.x_EP = eta / (2.*np.sqrt(self.k0*self.k1 * (1. + np.cos(theta))))
        self.y_EP = 0.0
    
    
    def k(self, n):
        """Return longitudinal wavevector."""
        return pi*np.sqrt(self.N**2 - n**2)
    
    def get_cycle_parameters(self, t):
        return self.x_EP, self.y_EP
    
    def H(self, t):
        """
        Return parametrically dependent Hamiltonian at time t,
        
            H = H(x(t), y(t)) .
            
        If x and y are specified directly, t is ignored and H(x,y) is
        returned instead.
        
            Parameters:
                t: float
                x, y: float, optional
                
            Returns:
                H: (2,2) ndarray
        """
        eps, delta = self.get_cycle_parameters(t)
        
        B = -1j*(np.exp(1j*self.theta_boundary) + 1) * \
                    self.kr/2. * np.sqrt(self.k0/(2.*self.k1))
        
        H11 = -self.k0 - 1j*self.eta/2.
        H12 = B*eps
        H21 = B.conj()*eps
        H22 = -self.k0 - delta - 1j*self.eta*self.k0/(2.*self.k1)
        
        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        return H
 
    def get_init_state(self):
        return np.array([1,1])/np.sqrt(2.)
        #return np.array([0,1])
        
    def get_boundary(self, x=None, eps=None, delta=None,
                     d=None, kr=None, theta_boundary=None):
        """Get boundary function xi."""
        
        # if variables not supplied set defaults
        if x is None:
            x = self.t
        if eps is None and delta is None:
            eps, delta = self.get_cycle_parameters(self.t)
        if d is None:
            d = self.d
        if kr is None:
            kr = self.kr
        if theta_boundary is None:
            theta_boundary = self.theta_boundary
        
        # reverse x-coordinate for backward propagation
        if self.loop_direction == '+':
            x = x[...,::-1]
        
        xi_lower = eps*np.sin((kr + delta)*x)
        xi_upper = d + eps*np.sin((kr + delta)*x + theta_boundary)
        
        return xi_lower, xi_upper

    
def generate_length_dependent_calculations(eta=0.1, eps_f=1.0, delta=0.0,
                                           L=100, N=1.01):
    """Prepare length dependent greens_code input for VSC calculations."""
    import os
    import shutil
    import fileinput
    
    pwd = os.getcwd()
    xml = "{}/input.xml".format(pwd)
    
    kr = (N - np.sqrt(N**2 - 1))*pi
    lambda0 = pi/(kr + delta)
    
    #for Ln in np.linspace(1,L,L):
    for Ln in np.arange(lambda0,L,lambda0):
        for loop_dir in '-':
            directory = "{}/eta_{}_L_{}_Ln_{}_{}".format(pwd, eta, L,
                                                         Ln, loop_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)
                
            params = {
                #'L': L,
                'L': Ln,
                'eta': eta,
                'N': N,
                'loop_direction': loop_dir,
                'loop_type': 'Varcircle',
                'init_loop_phase': 0.0,
                'init_state': 'a'
            }
            
            filename = ("N_{N}_{loop_type}_phase_{init_loop_phase:.3f}pi"
                        "_initstate_{init_state}_L_{L}_eta_{eta}_"
                        "{loop_direction}").format(**params).replace(".","")
            params['init_loop_phase'] *= pi
            
            WG = EP_Waveguide_EP_Trajectory(**params)
            WG.x_EP *= eps_f
            WG.y_EP += delta
            xi, _ = WG.get_boundary()
            
            # truncate x and xi arrays to reduced length Ln
            #N_file = int(WG.tN * Ln/L)
            #x = WG.t[:N_file]
            #xi = xi[:N_file]
            x = WG.t
            N_file = len(x)
            
            np.savetxt(filename + ".profile", zip(x, xi))
            shutil.copy(xml, directory)
                
            src_xml = open(xml)
            out_xml = open("{}/input.xml".format(directory), "w")
            
            replacements = {
                r'name="L">L':
                    r'name="L">{}'.format(Ln),
                r'name="N_file">N_file':
                    r'name="N_file">{}'.format(N_file),
                r'name="file">file':
                    r'name="file">{}.profile'.format(filename),
                r'name="Gamma0p_min">Gamma0p_min':
                    r'name="Gamma0p_min">{}'.format(eta),
                r'name="Gamma0p_max">Gamma0p_min':
                    r'name="Gamma0p_max">{}'.format(eta)
            }
            
            for line in src_xml:
                for src, target in replacements.iteritems():
                    line = line.replace(src, target)
                out_xml.write(line)
            src_xml.close()
            out_xml.close()
            
            os.chdir(pwd)

def length_scan_wrapper(eps_i=0.75, eps_f=1.25, eps_d=0.05,
                        delta_i=0.0, delta_f=0.1, delta_d=0.05):
    import os
    import shutil
    import fileinput
    
    pwd = os.getcwd()
    xml = "{}/input.xml".format(pwd)
    
    eps_EP_eff = 0.0297403          # N=1.01, d=1, eta=0.1, theta=0.0
    delta_EP_eff = 0.0
    
    for eps in np.arange(eps_i, eps_f, eps_d)*eps_EP_eff:
        for delta in np.arange(delta_i, delta_f, delta_d):
            
            directory = "{}/eps_{}_delta_{}".format(pwd, eps, delta)
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)
            shutil.copy(xml, directory)
            
            generate_length_dependent_calculations(eps=eps, delta=delta,
                                                   L=40, eta=0.1, N=1.01)
            
            os.chdir(pwd)


def get_psi():
    WG = EP_Waveguide_EP_Trajectory(L=40, eta=0.1, N=1.01)
    #WG.x_EP = 0.02
    x, _, _ = WG.solve_ODE()
    b0 = WG.Psi[:,0]
    b1 = WG.Psi[:,1]
    
    semilogy(x,abs(b0),"r-")
    semilogy(x,abs(b1),"g-")
    show()
    
    

if __name__ == '__main__':
    eps_EP = 0.0297403059911
    generate_length_dependent_calculations(N=1.01, eta=0.1, L=150,
                                           eps_f=1.17, delta=0.)
                                           #eps_f=1./2., delta=eps_EP*(1+np.sqrt(3)/2.))
    #generate_length_dependent_calculations(N=1.01, eta=0.1, L=50, eps=2.0, delta=0.0)
    #get_psi()
    #length_scan_wrapper()

    