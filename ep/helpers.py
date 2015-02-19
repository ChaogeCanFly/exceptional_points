from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from numpy import pi
from scipy.linalg import eig
from scipy.integrate import trapz


class FileOperations():
    """Simple  class to handle the output of class parameters to stdout and
    a .cfg file.

        ### possibly outdated -> json.dumps ###
    """

    def __init__(self, filename=None):
        filename = '{}.cfg'.format(filename)
        self.file = open(filename, 'a')

    def write(self, text):
        print text
        self.file.write(text)
        self.file.write("\n")

    def close(self):
        self.file.close()


def c_eig(H, left=False, **kwargs):
    """Wrapper for scipy.linalg.eig(H) that returns modified eigenvalues
    and eigenvectors.

    The returned vectors are normalized according to the biorthogonal product,

        <psi_l|phi_r> = delta_{psi,phi}

    with

        H|psi_r> = E|psi_r>, <psi_l|H = <psi_l|E ,

    instead of the standard (hermitian) inner product

        (|psi>.dagger) * |psi> = 1


        Parameters:
        -----------
            H:  (2,2) ndarray
                Hamiltonian matrix
            left: bool (default: False)
                Whether to calculate left eigenvectors as well

        Returns:
        --------
          eigenvalues:  (2,)   ndarray
          left eigenvectors:  (2,2)  ndarray
          right eigenvectors: (2,2)  ndarray
    """

    # get eigenvalues and eigenvalues of matrix H
    # multiple possibilities:
    # 1) from option left=True
    eVals, eVecs_l, eVecs_r = eig(H, left=True, **kwargs)
    eVecs_l = eVecs_l.conj()
    #print np.einsum('ij,ij -> i', eVecs_r.conj(), eVecs_r)
    #print np.einsum('ij,ij -> i', eVecs_l.conj(), eVecs_l)
    #test_eigenvalues(eVals, eVecs_l, eVecs_r, H)

    # 2) left eigenvalues are the (transposed) right eigenvectors of H.T
    # (results in extremely large eigenvectors)
    #eVals, eVecs_r = eig(H, **kwargs)
    #_, eVecs_l = eig(H.T, **kwargs)
    #test_eigenvalues(eVals, eVecs_l, eVecs_r, H)

    # 3) X_L * X_R = 1, X_L = inv(X_R).T
    #eVals, eVecs_r = eig(H, **kwargs)
    #eVecs_l = inv(eVecs_r).T
    #test_eigenvalues(eVals, eVecs_l, eVecs_r, H)

    # normalize eigenvectors w.r.t. biorthogonality
    # scipy.linalg.eig uses LAPACK's geev routine to compute the eigensystem,
    # which returns eigenvectors that are "normalized to have Euclidian norm
    # equal to 1 and largest component real" [1]
    # [1] https://software.intel.com/en-us/node/521147
    c_norm = lambda ev_l, ev_r: np.sqrt(ev_l.dot(ev_r))

    for n in (0, 1):
        # here one has freedom to have N = N_r * N_l, i.e., with
        # N_n = rho_n * exp(i*phi_n):
        #   rho = rho_r * rho_l and
        #   phi = phi_r + phi_l
        # we choose to leave the Euclidian norm of the right eigenvectors
        # intact, s.t. abs(eVecs_r)**2 = 1
        # and put the remaining normalization into the left eigenvectors
        N = c_norm(eVecs_l[:,n],eVecs_r[:,n])
        eVecs_l[:,n] /= N**2

    #print "after R:", np.einsum('ij,ij -> i', eVecs_r.conj(), eVecs_r)
    #print "after L:", np.einsum('ij,ij -> i', eVecs_l.conj(), eVecs_l)

    if left:
        return eVals, eVecs_l, eVecs_r
    else:
        return eVals, eVecs_r


def c_trapz(f, dx, **kwargs):
    """Wrapper for scipy.integrate.trapz that allows to integrate complex-valued
    arrays.

        Parameters:
        -----------
            f:  (N,) ndarray
           dx:  float

        Returns:
        --------
            c_trapz: (N,) ndarray
    """

    real_int = trapz(f.real, dx=dx, **kwargs)
    imag_int = trapz(f.imag, dx=dx, **kwargs)

    return real_int + 1j*imag_int


def c_gradient(f, dx):
    """Wrapper for numpy.gradient that allows to calculate gradients for complex-
    valued arrrays.

        Parameters:
        -----------
            f: (N,) ndarray
            dx: float

        Returns:
        --------
            c_gradient: (N,) ndarray
    """

    real_grad = np.gradient(f.real, dx)
    imag_grad = np.gradient(f.imag, dx)

    return real_grad + 1j*imag_grad


def map_trajectory(c1, c2, E1, E2):
    """Function to determine the trajectory's character based on the amplitudes
    c1 and c2.

        Parameters:
        -----------
            c1, c2: ndarray
                Amplitudes c1 and c2.
            E1, E2: ndarray
                Real or imaginary parts of the energies E1 and E2.

        Returns:
        --------
            mapped trajectory: ndarray
    """
    c1, c2 = [ np.abs(x)**2 for x in c1, c2 ]

    return (E1*c1 + E2*c2)/(c1 + c2)


def set_scientific_axes(ax, axis='x'):
    """Set axes to scientific notation."""

    #xticks(rotation=30)
    #ax.ticklabel_format(style='sci', axis=axis, scilimits=(0,0), useOffset=False)
    ax.ticklabel_format(style='plain', axis=axis, useOffset=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))


def cmap_discretize(cmap, indices):
    """Discretize colormap according to indices list.

        Parameters:
        -----------
            cmap: str or Colormap instance
            indices: list

        Returns:
        --------
            segmented colormap
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    indices = np.ma.concatenate([[0],indices,[1]])
    N = len(indices)

    colors_i = np.concatenate((np.linspace(0,1.,N),
                               (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)

    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki],
                        colors_rgba[i,ki]) for i in xrange(N) ]

    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def test_eigenvalues(eVals, eVecs_l, eVecs_r, H):
    """Test eigenvalue problem.

    The output contains both the eigenvalues and eigenvectors (left and right),
    as well as the left and right hand sides of the eigenvalue equations

        H v_i = E_i v_i
    or
        v_i H = E_i v_i

    respectively.

        Parameters:
        -----------
            eVals: (2,) ndarray
            eVecs_l: (2,2) ndarray
            eVecs_r: (2,2) ndarray
            H: (2,2) ndarray

        Returns:
        --------
            None
    """

    print 50*"#"
    print
    print "eVals\n ", eVals
    print "eVecs_r\n", eVecs_r
    print "eVecs_l\n", eVecs_l
    print

    for n in (0,1):
        print "ev_l*H\n  ", eVecs_l[:,n].dot(H)
        print "e1*ev_l\n  ", eVals[n]*eVecs_l[:,n]
        print
        print "H*ev_r\n  ", H.dot(eVecs_r[:,n])
        print "e1*ev_r\n  ", eVals[n]*eVecs_r[:,n]
        print

    print 50*"#"


def get_height_profile(X, Y, sigma_x=1e-4, rho_y=1e-2):
        """Return customized height profile.

            Parameters:
            -----------
                X, Y: (Nx,Ny) ndarray
                    Geometry meshgrid.
                sigma_x, rho_y: float
                    Standard deviation in x-, and smearing in y direction.

            Returns:
            --------
                W_Gauss_Fermi: (Nx,Ny) ndarray
                    Height-profile with Gauss in x-, and Fermi shape in y direction.
                W_Fermi: (Nx,Ny) ndarray
                    Height profile with Fermi shape in y direction.
                wmax, wmin: float
                    Maximum and minimum of the height profile.
        """

        W_Gauss_Fermi, W_Fermi = 0.*X, 0.*X

        W_Gauss_Fermi = np.exp(-(X-X.mean())**2/(2*sigma_x))
        W_Gauss_Fermi /= np.sqrt(2*pi*sigma_x**2)
        W_Gauss_Fermi *= 1./(np.exp(-(Y-Y.mean())/rho_y) + 1)

        W_Fermi = 1./(np.exp(-(X-X.mean())/rho_y) + 1)
        W_Fermi = W_Fermi*W_Gauss_Fermi.max()/W_Fermi.max()

        wmax = W_Fermi.max()
        wmin = W_Fermi.min()

        return W_Gauss_Fermi, W_Fermi, wmax, wmin


def get_local_peaks(v, peak_type='minimum'):
    """Find the local maxima of a multidimensional array v.
    Returns a boolean mask."""
    # taken from stackoverflow.com/questions/3684484

    from scipy.ndimage.filters import minimum_filter, maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

    if peak_type == 'minimum':
        peak_filter = minimum_filter
    elif peak_type == 'maximum':
        peak_filter = maximum_filter

    neighborhood = generate_binary_structure(2,2)
    local_peak = peak_filter(v, footprint=neighborhood) == v
    background = (v == 0)
    eroded_background = binary_erosion(background,
                                       structure=neighborhood,
                                       border_value=1)
    detected_peaks = local_peak - eroded_background

    return detected_peaks

def get_local_minima(v):
    """Find the local maxima of a multidimensional array v.
    Returns a boolean mask."""
    # taken from stackoverflow.com/questions/3986345
    return ((v <= np.roll(v,  1, 0)) &
            (v <= np.roll(v, -1, 0)) &
            (v <= np.roll(v,  1, 1)) &
            (v <= np.roll(v, -1, 1)))


if __name__ == '__main__':
    pass
