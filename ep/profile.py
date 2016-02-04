#!/usr/bin/env python2.7


import json
import numpy as np
from numpy import pi
import os
import shutil
import subprocess
import sys

from ep.waveguide import Neumann, Dirichlet, DirichletPositionDependentLoss
import helper_functions


class Generate_Profiles(object):
    """A class to prepare length dependent greens_code input for VSC
    calculations.

        Parameters:
        -----------
            eta: float
                Dissipation coefficient.
            L:   float
                Waveguide length.
            N:   float
                Number of open modes int(k*W/pi).
            loop_type: str
                Specifies path in (epsilon,delta) parameter space.
            loop_direction: str ("-"|"+")
                Loop direction around the EP.
            init_phase: float
                Starting angle on parameter trajectory.
            theta: float
                Phase difference bewteen upper and lower boundary (in multiples
                of pi).
            eps_factor: float
                Constant to shift x_R0 -> x_EP * eps_factor.
            eps: float
                Set value for x_R0 to eps.
            delta: float
                Constant to set y_EP (or, equivalently, y_EP -> y_EP + delta).
            full_evolution: bool
                Whether to build intermediate waveguide boundaries with x < L.
            input_xml: str
                Input xml file to be supplied with length-dependent data.
            pphw: int
                Points per half wavelength (determines grid-spacing).
            nx_part: int
                Parts into which the Border Hamiltonian rectangle is divided
                into.
            custom_directory: str
                Custom directory into which to copy the .xml and .profile
                files.
            neumann: bool
                Whether to use Neumann boundary conditions.
            use_variable_length: bool
                Whether to use a multiple of the wavelength for the system
                size.
            smearing: bool
                Return a profile which is smeared out at the edges.
            heatmap: bool
                Whether to calculate a (eta,L) heatmap.

        Attributes:
        -----------
            cwd: str
                Current working directory.
            WG: Waveguide class object

        Notes:
        -----
            The waveguide boundary is prepared such that the length is an
            integer multiple of the detuned resonant wavelength,
            2*pi/(kr + delta).

    """
    def __init__(self, eps_factor=1.0, eps=None, delta=0.0,
                 full_evolution=False, input_xml="input.xml", pphw="200",
                 nx_part="50", custom_directory=None, neumann=0,
                 use_variable_length=False, smearing=False,
                 heatmap=False, **waveguide_kwargs):

        self.waveguide_kwargs = waveguide_kwargs
        # make waveguide_kwargs available in class namespace
        self.__dict__.update(waveguide_kwargs)

        self.eps = eps
        self.eps_factor = eps_factor
        self.delta = delta
        self.full_evolution = full_evolution
        self.input_xml = input_xml
        self.xml = os.path.abspath(input_xml)
        self.pphw = pphw
        self.nx_part = nx_part
        self.custom_directory = custom_directory
        self.neumann = neumann
        self.use_variable_length = use_variable_length
        self.smearing = smearing
        self.heatmap = heatmap

        self.cwd = os.getcwd()

        if heatmap:
            self._heatmap()
        else:
            self._length()

    def _heatmap(self):
        """Generate a heatmap in (eta, L) space."""

        if len(self.L) == 3:
            L_range = np.arange(*self.L)
        else:
            L_range = [self.L[0]]
        print "L_range", L_range

        if len(self.eta) == 3:
            eta_range = np.arange(*self.eta)
        else:
            eta_range = [self.eta[0]]
        print "eta_range", eta_range

        for Ln in L_range:
            for eta_n in eta_range:
                params = {'L': Ln,
                          'eta': eta_n}
                print "Warning: check length values in the .xml and .profile files!"
                self.L = Ln
                self.eta = eta_n
                self.waveguide_kwargs.update(**params)
                self._length()

    def _length(self):
        """Generate length-dependent input-files for VSC runs."""

        # calculate the waveguide data
        if self.neumann:
            WG = Neumann(**self.waveguide_kwargs)
        else:
            WG = Dirichlet(**self.waveguide_kwargs)
        self.WG = WG

        if self.eps:
            self.WG.x_R0 = self.eps
        else:
            self.WG.x_R0 = self.WG.x_EP * self.eps_factor
        self.WG.y_EP += self.delta

        if self.use_variable_length:
            lambda0 = np.abs(pi/(self.WG.kr + self.delta))
            L_range = np.arange(lambda0, self.L, 2*lambda0)
        else:
            L_range = np.linspace(1, self.L, self.L)

        if not self.full_evolution:
            L_range = L_range[-1:]

        for Ln in L_range:
            self.Ln = Ln

            ID_params = {'Ln': Ln}
            ID_params.update(**self.waveguide_kwargs)
            ID = ("N_{N}_t_{loop_type}_phase_{init_phase:.3f}_L_{L}_Ln_{Ln:.3f}"
                  "_eta_{eta}_direction_{loop_direction}").format(**ID_params)
            self.filename = ID

            if self.custom_directory:
                self.directory = os.path.abspath(self.custom_directory)
            else:
                self.directory = os.path.abspath(ID)

            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            os.chdir(self.directory)

            # print profile properties to file
            with open("EP_SETTINGS.cfg", "w") as f:
                d = {key: value for key, value in vars(self.WG).items()
                        if not (isinstance(value, np.ndarray) or
                                isinstance(value, complex) or
                                isinstance(value, type(lambda x: 1)))}
                data = json.dumps(d, sort_keys=True, indent=4)
                f.write(data)

            # print epsilon/delta values
            try:
                x_eps, y_delta = self.WG.get_cycle_parameters(self.WG.t)
                # save some disk space
                x_eps, y_delta = [ x[::25] for x in x_eps, y_delta ]
                np.savetxt(self.filename + ".eps_delta", zip(x_eps, y_delta))
            except:
                print "Warning: cannot write .eps_delta file"

            # print profile
            # make sure that N_file = r_nx
            nyout = self.N*self.pphw
            dx = 1./(nyout + 1.)
            self.r_nx = int(Ln/dx)
            x = np.linspace(0, Ln, self.r_nx)
            print "r_nx:", self.r_nx

            xi_lower, xi_upper = self.WG.get_boundary(x=x, smearing=self.smearing)
            np.savetxt(self.filename + ".upper_profile", zip(x, xi_upper))
            np.savetxt(self.filename + ".lower_profile", zip(x, xi_lower))
            
            # write to xml and return to cwd
            self._copy_and_replace(self.xml)
            os.chdir(self.cwd)

    def _copy_and_replace(self, infile):
        """Copy the input file and write to output file with replaced
        values."""

        shutil.copy(infile, self.directory)

        N_file = self.r_nx
        file_upper = self.filename + ".upper_profile"
        file_lower = self.filename + ".lower_profile"

        # replacements = {
        #         'modes"> modes':     'modes"> {}'.format(self.N),
        #         'halfwave"> pphw':   'halfwave"> {}'.format(self.pphw),
        #         'L"> L':             'L"> {}'.format(self.Ln),
        #         'N_file"> N_file':   'N_file"> {}'.format(N_file),
        #         'file"> file_upper': 'file"> {}'.format(file_upper),
        #         'file"> file_lower': 'file"> {}'.format(file_lower),
        #         'nx_part"> nx_part': 'nx_part"> $r_nx/{}'.format(self.nx_part),
        #         'neumann"> neumann': 'neumann"> {}'.format(self.neumann),
        #         'Gamma0"> Gamma0':   'Gamma0"> {:f}'.format(self.eta)
        #         }

        replacements = {
                'NAME':  'sine_boundary',
                'MODES':  '{}'.format(self.N),
                'PPHW':   '{}'.format(self.pphw),
                'LENGTH': '{}'.format(self.Ln),
                'WIDTH': '1.0',
                'N_FILE_BOUNDARY': '{}'.format(N_file),
                'NEUMANN': '{}'.format(self.neumann),
                'BOUNDARY_UPPER': '{}'.format(file_upper),
                'BOUNDARY_LOWER': '{}'.format(file_lower),
                'GAMMA0': '{:f}'.format(self.eta)
                }

        out_xml = os.path.abspath("input.xml")

        with open(self.xml) as src_xml:
            src_xml = src_xml.read()

        for src, target in replacements.iteritems():
            src_xml = src_xml.replace(src, target)

        out_xml = os.path.abspath("input.xml")
        with open(out_xml, "w") as out_xml:
            out_xml.write(src_xml)


def parse_arguments():
    """Parse input for function Generate_Profiles(*args, **kwargs).

        Returns:
        --------
            parse_args: dict
    """
    import json
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter as help_formatter

    parser = argparse.ArgumentParser(formatter_class=help_formatter)

    parser.add_argument("--eta", nargs="+", default=[0, 1, 1], type=float,
                        help="Dissipation coefficient (eta_min eta_max eta_incr)")
    parser.add_argument("-L", nargs="+", default=[100, 101, 1], type=float,
                        help="Waveguide length (L_min L_max L_incr)")
    parser.add_argument("--N", nargs="?", default=1.05, type=float,
                        help="Number of open modes int(k*d/pi)")
    parser.add_argument("-t", "--loop-type", default="Bell", type=str,
                        help="Specifies path in (eps,delta) parameter space")
    parser.add_argument("-o", "--loop-direction", default="-", type=str,
                        help="Loop direction around the EP")
    parser.add_argument("--init-phase", default=0.0, type=float,
                        help="Starting angle on parameter trajectory")
    parser.add_argument("-T", "--theta", default=0.0, type=float,
                        help=("Phase difference between upper and lower "
                              "boundary (in multiples of pi)"))
    parser.add_argument("--eps-factor", nargs="?", default=1.0, type=float,
                        help="Constant to shift x_R0 -> x_EP * eps_factor")
    parser.add_argument("--eps", nargs="?", default=None, type=float,
                        help="Set value for x_R0 to eps (only if not None)")
    parser.add_argument("-d", "--delta", nargs="?", default=0.0, type=float,
                        help=("Constant to set y_EP (or, equivalently, "
                              "y_EP -> y_EP + delta)"))
    parser.add_argument("-f", "--full-evolution", action="store_true",
                        help=("Whether to build intermediate waveguide "
                              "boundaries with x < L"))
    parser.add_argument("-i", "--input-xml", default="input.xml", type=str,
                        help=("Input xml file to be supplied with length-"
                              "dependent data"))
    parser.add_argument("-p", "--pphw", default=200, type=int,
                        help=("Points per half wavelength (determines grid-"
                              "spacing)"))
    parser.add_argument("-x", "--nx-part", default=100, type=int,
                        help=("Parts into which the Border Hamiltonian "
                              "rectangle is divided into"))
    parser.add_argument("-c", "--custom-directory", default=None, type=str,
                        help=("Custom directory into which to copy the .xml "
                              "and .profile files."))
    parser.add_argument("-n", "--neumann", default=1, type=int,
                        help="Whether to use Neumann boundary conditions.")
    parser.add_argument("-u", "--use-variable-length", action="store_true",
                        help=("Whether to use a multiple of the wavelength for "
                              "the system size."))
    parser.add_argument("-s", "--smearing", action="store_true",
                        help=("Return a profile which is smeared out at "
                              "the edges."))
    parser.add_argument("-H", "--heatmap", action="store_false",
                        help="Whether to NOT calculate a (eta,L) heatmap.")

    parse_args = vars(parser.parse_args())

    # add git log output
    parse_args.update({"git log": helper_functions.get_git_log(relative_git_path="..")})

    print json.dumps(parse_args, sort_keys=True, indent=4)

    with open("EP_PARSE_SETTINGS.cfg", "w") as f:
        data = json.dumps(parse_args, sort_keys=True, indent=4)
        f.write(data)

    # remove git log output
    parse_args.pop("git log")

    return parse_args


if __name__ == '__main__':
    Generate_Profiles(**parse_arguments())
