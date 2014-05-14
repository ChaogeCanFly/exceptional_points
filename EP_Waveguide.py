#!/usr/bin/env python2.7

from EP_Base import *
import numpy as np
from numpy import pi

class EP_Waveguide(EP_Base):
    """EP_Waveguide class."""
    
    def __init__(self, L=100, d=1.0, eta=0.05, N=1.5, theta=0.0, **kwargs):
        """Exceptional Point (EP) waveguide class.
        
        Copies methods and variables from EP_Base class and adds new parameters.
        
            Additional parameters:
            ----------------------
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
        #self.position_dependent_eta = False         # use pos. dep. loss
        self.N = N                                  # number of open modes
        self.k0, self.k1 = self.k(0), self.k(1)     # longitudinal wavenumbers
                                                    # for mode n=0 and n=1
        self.kr = self.k0 - self.k1                 # wavenumber difference
        
        self.theta_boundary = theta                 # phase angle between upper
                                                    # and lower boundary
        self.x_EP = eta / (2.*np.sqrt(self.k0*self.k1 * (1. + np.cos(theta))))
        self.y_EP = 0.0
        
        # change initial conditions
        self.x_R0 = self.x_EP     # circling radius
        self.y_R0 = self.x_EP     # circling radius
    
    
    def k(self, n):
        """Return longitudinal wavevector."""
        return pi*np.sqrt(self.N**2 - n**2)
    
    #def eta_x(self, x):
    #    """Return position dependent dissipation coefficient."""
    #    return self.eta * np.sin(pi/self.T * x)
    
    def H(self, t, eps=None, delta=None):
        """
        Return parametrically dependent Hamiltonian at time t,
            H = H(x(t), y(t)).
        If x and y are specified directly, t is ignored and H(x,y) is
        returned instead.
        
            Parameters:
            -----------
                t: float
                x, y: float, optional
                
            Returns:
            --------
                H: (2,2) ndarray
                
        """
        
        if eps is None and delta is None:
            eps, delta = self.get_cycle_parameters(t)
            
        B = (-1j * (np.exp(1j*self.theta_boundary) + 1) * 
                      self.kr/2. * np.sqrt(self.k0/(2.*self.k1)))
        
        H11 = -self.k0 - 1j*self.eta/2.
        H12 = B*eps
        H21 = B.conj()*eps
        H22 = -self.k0 - delta - 1j*self.eta*self.k0/(2.*self.k1)
        
        H = np.array([[H11, H12],
                      [H21, H22]], dtype=complex)
        
        return H
  
  
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
        
        x_EP, x_R0 = self.x_EP, self.x_R0
        y_EP, y_R0 = self.y_EP, self.y_R0
        w = self.w
        phi0 = self.init_loop_phase
        loop_type = self.loop_type
            
        if loop_type == "Circle":
            lambda1 = lambda t: x_EP + x_R0*np.cos(w*t + phi0)
            lambda2 = lambda t: y_EP + y_R0*np.sin(w*t + phi0)
            return lambda1(t), lambda2(t)
        
        elif loop_type == "Varcircle":
            y_EP = 0.0
            lambda1 = lambda t: x_EP - x_EP*np.cos(w*t)
            lambda2 = lambda t: y_EP - x_EP*np.sin(w*t) + phi0
            return lambda1(t), lambda2(t)
        
        elif loop_type == "Bell":
            if self.loop_direction == '+':
                a = -1
            else:
                a = +1
            lambda1 = lambda t: x_EP * (1. - np.cos(w*t))
            lambda2 = lambda t: 20. * x_EP * (w*t/pi - a) + phi0
            return lambda1(t), lambda2(t)
        
        elif loop_type == "Constant":
            return x_EP, y_EP
        
        elif loop_type == "Constant_delta":
            return x_EP * (1.0 - np.cos(w*t)), y_EP
        
        else:
            raise Exception(("Error: loop_type {}"
                             "does not exist!").format(loop_type))
        
        
    def sample_H(self, xN=None, yN=None):
        """Sample local eigenvalue geometry of H.
        
            Parameters:
            -----------
                xN, yN: int
                    Number of sampling points.
                    
            Returns:
            --------
                X, Y: (N,N) ndarray
                Z: (N,N,2) ndarray
        
        """
        
        if xN is None:
            xN = 5*10**2
        if yN is None:
            yN = xN
        
        x = np.linspace(self.x_EP - 1.1*self.x_R0,
                        self.x_EP + 1.1*self.x_R0, xN)
        y = np.linspace(self.y_EP - 1.1*self.y_R0,
                        self.y_EP + 1.1*self.y_R0, yN)
        
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((xN,yN,2), dtype=complex)
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                Z[i,j,:] = c_eig(self.H(0,xi,yj))[0]
                
        # circumvent indexing='ij' option in np.meshgrid
        return X.T, Y.T, Z


    def draw_wavefunction(self):
        """Plot wavefunction."""
        
        x, b0, b1 = self.t, self.phi_a, self.phi_b
        yN = len(x)/self.T
        y = np.linspace(-0.1,self.d+0.1,yN)
        
        def phi(x,y):
            phi = b0 + b1 * (np.sqrt(2.*self.k0/self.k1) *
                              np.cos(pi*y)*np.exp(-1j*self.kr*x))
            return phi

        X, Y = np.meshgrid(x,y)
        Z = abs(phi(X,Y))**2
        
        p = pcolormesh(X,Y,Z)
        #cb = colorbar(p)
        #cb.set_label("Wavefunction")
        
        
    def draw_dissipation_coefficient(self, cax=None):
        """Plot position dependent dissipation coefficient."""
        
        x, b0, b1 = self.t, self.phi_a, self.phi_b
        y = np.linspace(-0.1,self.d+0.1,2)
        
        X, Y = np.meshgrid(x,y)
        Z = self.eta_x(X)
        
        import brewer2mpl as brew
        bmap = brew.get_map('YlOrRd',
                            'sequential', 9).mpl_colormap
        p = pcolormesh(X,Y,Z)
        p.cmap = bmap
        cb = colorbar(p, ax=cax)
        cb.set_label("Loss")
    
    
    def get_boundary(self, x=None, eps=None, delta=None,
                     d=None, kr=None, theta_boundary=None):
        """Get the boundary function xi as a function of the spatial coordinate x.
        
            Parameters:
            -----------
                x: ndarray
                    Spatial/temporal coordinate.
                eps: float
                    Boundary roughness strength.
                delta: float
                    Boundary frequency detuning.
                d: float
                    Waveguide width.
                kr: float
                    Boundary modulation frequency.
                theta_boundary: float
                    Phase difference between lower and upper boundary.
            
            Returns:
            --------
                xi_lower: float
                    Lower boundary function.
                xi_upper: float
                    Upper boundary function.
                    
        """
        
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
    
    
    def get_boundary_contour(self, X, Y):
        """Get the boundary contour."""
        
        lower, upper = self.get_boundary(X)
        mask_upper = Y > upper
        mask_lower = Y < lower
        Z = mask_upper + mask_lower
        
        return X, Y, Z
    
    
    def draw_boundary(self):
        """Draw the boundary profile."""
        
        x = self.t #self.t[::2]
        #eps, delta = self.get_cycle_parameters(x)
        
        yN = len(x)/self.T
        y = np.linspace(-0.1, self.d+0.1, yN)

        X, Y = np.meshgrid(x, y)
        X, Y, Z = self.get_boundary_contour(X, Y)
        
        contourf(X, Y, Z, [0.9,1], colors="k")
        return X, Y, Z

    
def generate_profile_heatmap():
    """Generate a matrix of (eta, L) values for VSC calculations."""
    
    import os
    import shutil
    import fileinput
    
    pwd = os.getcwd()
    xml = "{}/input.xml".format(pwd)
    
    for eta in np.arange(0.01, 0.055, 0.005)/100:
        for L in np.arange(60, 150, 10):
            for loop_dir in '-', '+':
                directory = "{}/eta_{}_L_{}_{}".format(pwd, eta,
                                                       L, loop_dir)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                os.chdir(directory)
                    
                params = {
                    'L': L,
                    'eta': eta,
                    'N': 1.21,
                    'init_loop_phase': 0.001,
                    'loop_direction': loop_dir,
                    'loop_type': 'Varcircle',
                    'init_state': 'a'
                }
                
                filename = ("N_{N}_{loop_type}_phase_{init_loop_phase:.3f}pi"
                            "_initstate_{init_state}_L_{L}_eta_{eta}_"
                            "{loop_direction}").format(**params).replace(".","")
                params['init_loop_phase'] *= pi
                
                WG = EP_Waveguide(**params)
                xi, _ = WG.get_boundary()

                np.savetxt(filename + ".profile", zip(WG.t, xi))
                
                shutil.copy(xml, directory)
                N_file = WG.tN
                    
                src_xml = open(xml)
                out_xml = open("{}/input.xml".format(directory), "w")
                
                replacements = {
                    r'name="L">L':
                        r'name="L">{}'.format(L),
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

    
def generate_length_dependent_calculations(eta=0.3, L=100, N=1.01,
                                           init_loop_phase=0.0,
                                           loop_type="Varcircle",
                                           loop_direction="-",
                                           eps_factor=1.0, eps=None,
                                           delta=0.0, 
                                           full_evolution=False,
                                           write_cfg=True,
                                           input_xml="input.xml",
                                           pphw="200", r_nx_part="50",
                                           custom_directory=None):
    """
    Prepare length dependent greens_code input for VSC calculations. The waveguide
    boundary is prepared such that the length is an integer multiple of the detuned
    resonant wavelength, 2*pi/(kr + delta).
    
        Parameters:
        -----------
            eta: float
                Dissipation coefficient.
            L:   float
                Waveguide length.
            N:   float
                Number of open modes int(k*d/pi).
            loop_type: str
                Specifies path in (epsilon,delta) parameter space.
            init_loop_phase: float
                Starting angle on parameter trajectory.
            eps_factor: float
                Constant to shift x_EP -> x_EP * eps_factor.
            eps: float
                Set value for x_EP to eps (only done if set_x_EP=True).
            delta: float
                Constant to set y_EP (or, equivalently, y_EP -> y_EP + delta).
            full_evolution: bool
                Whether to build intermediate waveguide boundaries with x < L.
            write_cfg: bool
                Whether to write WG class attributes to file.
            input_xml: str
                Input xml file to be supplied with length-dependent data.
            pphw: int
                Points per half wavelength (determines grid-spacing).
            r_nx_part: int
                Parts into which the Border Hamiltonian rectangle is divided into.
            custom_directory: str
                Custom directory into which to copy the .xml and .profile files.
        
        Returns:
        --------
            None
    
    """
    import os
    import shutil
    import fileinput
    import json
    import re
    
    pwd = os.getcwd()
    xml = "{}/{}".format(pwd, input_xml)
    
    #if delta is None:
    #    L_range = np.linspace(1,L,L)
    #else:
    kr = (N - np.sqrt(N**2 - 1))*pi
    lambda0 = pi/(kr + delta)
    L_range = np.arange(lambda0,L,lambda0)
        
    if not full_evolution:
        L_range = (L_range[-1],)
    
    for Ln in L_range:
        params = {
            'L': Ln,
            'eta': eta,
            'N': N,
            'init_loop_phase': init_loop_phase,
            'loop_direction': loop_direction,
            'loop_type': loop_type,
        }
        
        if custom_directory:
            directory = "{}/{}".format(pwd, custom_directory)
        else:
            directory = "{}/eta_{eta}_L_{}_Ln_{}_{loop_direction}".format(pwd, L, Ln,
                                                                          **params)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        os.chdir(directory)
        
        filename = ("N_{N}_{loop_type}_phase_{init_loop_phase:.3f}pi"
                    "_L_{L}_eta_{eta}_"
                    "{loop_direction}").format(**params).replace(".","")
        params['init_loop_phase'] *= pi
        
        WG = EP_Waveguide(**params)
        
        if eps:
            WG.x_EP = eps
        else:
            WG.x_EP *= eps_factor
            
        WG.y_EP += delta
        xi, _ = WG.get_boundary()
        
        # print discretization data
        #with open(xml) as x:
        #    for line in x.readlines():
        #        if "points_per_halfwave" in line:
        #            pph = re.split("[><]", line)[-3]
        #            dx = 1./(float(pph) + 1)
        #            print
        #            print "Number of gridpoints per amplitude:", int(2.*eps/dx)
        #            print
                
        # print wire properties to file
        if write_cfg:
            with open("EP_SETTINGS.cfg", "w") as f:
                d = { key: value for key, value in vars(WG).items()
                      if not isinstance(value, np.ndarray)}
                data = json.dumps(d, sort_keys=True, indent=-1)
                f.write(data)

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
            r'halfwave">pphw': r'halfwave">{}'.format(pphw),
            r'"L">L': r'"L">{}'.format(Ln),
            r'"N_file">N_file': r'"N_file">{}'.format(N_file),
            r'"file">file': r'"file">{}.profile'.format(filename),
            r'"nx_part">$r_nx/50': r'"nx_part">$r_nx/{}'.format(r_nx_part),
            r'"Gamma0p_min">Gamma0p_min': r'"Gamma0p_min">{}'.format(eta),
            r'"Gamma0p_max">Gamma0p_min': r'"Gamma0p_max">{}'.format(eta)
        }
        
        for line in src_xml:
            for src, target in replacements.iteritems():
                line = line.replace(src, target)
            out_xml.write(line)
        src_xml.close()
        out_xml.close()
        
        os.chdir(pwd)

def parse_arguments():
    """
    Parse input for function generate_length_dependent_calculations(*args, **kwargs).
        
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
    parser.add_argument("-t", "--loop-type", default="Constant_delta", type=str,
                        help="Specifies path in (epsilon,delta) parameter space" )
    parser.add_argument("--init-loop-phase", default=0.0, type=float,
                        help="Starting angle on parameter trajectory" )
    parser.add_argument("--eps-factor", nargs="?", default=1.0, type=float,
                        help="Constant to shift x_EP -> x_EP * eps_factor" )
    parser.add_argument("--eps", nargs="?", default=None, type=float,
                        help="Set value for x_EP to eps (only done if not None)" )
    parser.add_argument("-d", "--delta", nargs="?", default=0.0, type=float,
                        help="Constant to set y_EP (or, equivalently, y_EP -> y_EP + delta)" )
    parser.add_argument("-f", "--full-evolution", action="store_true",
                        help="Whether to build intermediate waveguide boundaries with x < L")
    parser.add_argument("-w", "--write-cfg", action="store_false",
                        help="Whether to NOT write WG class attributes to file")
    parser.add_argument("-i", "--input-xml", default="input.xml", type=str,
                        help="Input xml file to be supplied with length-dependent data")
    parser.add_argument("-p", "--pphw", default=200, type=int,
                        help="Points per half wavelength (determines grid-spacing)")
    parser.add_argument("-r", "--r-nx-part", default=100, type=int,
                        help="Parts into which the Border Hamiltonian rectangle is divided into")
    parser.add_argument("-c", "--custom-directory", default=None, type=str,
                        help="Custom directory into which to copy the .xml and .profile files.")
    
    #subparsers = parser.add_subparsers(help='sub-command help')
    
    args = parser.parse_args()
    kwargs = vars(args)
    
    print json.dumps(kwargs, indent=4)
    if args.write_cfg:
        with open("EP_PARSE_SETTINGS.cfg", "w") as f:
            data = json.dumps(kwargs, sort_keys=True, indent=-1)
            f.write(data)    
    return kwargs
    
   
if __name__ == '__main__':
    
    generate_length_dependent_calculations(**parse_arguments())
    
    #generate_length_dependent_calculations(eta=0.0, L=100, 
    #                                       eps_factor=1.0, delta=0.3,
    #                                       set_x_EP=True, eps=0.01, 
    #                                       full_evolution=False,
    #                                       loop_type="Constant_delta")
