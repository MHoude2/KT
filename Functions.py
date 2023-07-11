"""Functions to produce JSA and moments for variable domain configs in (k,t) space"""

import numpy as np
from scipy.linalg import expm

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def FtS(domain,pump,z_list,k,t):
    """Generates the fourier transform evaluated at (k,t) of the product of the domain configuration 
    and pump pulse as functions of (z,t)

    Args:
        domain (array): array of +/- 1's characterizing the domain configuration
        pump (function): functional form of pump pulse as a function of (z,t) normalized to Np
        z_list (array): list of position values, taken at center point, of crystal positions
        k (array): array momentum values
    Returns:
        (array): S(k+k',t) matrix for a specific time 't' 
    """
    dz = z_list[1]-z_list[0]
    Expfac = np.exp(-1j*np.tensordot(k+k[:,np.newaxis],z_list,axes=0))/np.sqrt(2*np.pi)*dz
    Prod = Expfac*pump(z_list,t)*domain
    return np.sum(Prod,axis=2)


def Total_prop(domain,pump,z_list,k,t,ws,wi):
    """Generates the total Heisenberg propagator

    Args:
        domain (array): array of +/- 1's characterizing the domain configuration
        pump (function): functional form of pump pulse as a function of (z,t) normalized to Np
        z_list (array): list of position values, taken at center point, where nonlinearity exists
        k (array): array of momentum values
        ws (array): dispersion relation for signal photons (function of momenta)
        wi (array): dispersion relation for idler photon (function of momenta)
    Returns:
        (array): K(k+k',tf), the total Heisenberg propagator at final time 
    """
    #Initializing
    K = np.identity(2 * len(k), dtype=np.complex128)
    dk = k[1]-k[0]
    dt = t[1]-t[0]
    
    #Constructing the diagonal blocks
    Rs = np.diag(-1j*ws)
    Ri = np.diag(1j*wi)
    
    for i in t:
        S = 1j*FtS(domain,pump,z_list,k,i)*dk/np.sqrt(2*np.pi)
        Q = np.block([[Rs,S],[np.conjugate(S),Ri]])
        K = expm(Q*dt)@K

    return K

def JSA(K,dk):
    """Given a total Heisenberg propagator in (k,t) generates the JSA as well as any relevant
       moment matrix and property
    
    Args:
        K (array): Total Heisenberg propagator
        dk (float): momentum stepsize
    
    Returns:
        (array, :Joint spectral amplitude
        float,  :Number of signal photons
        float,  :Schmidt number
        array,  :M moment matrix
        array,  :signal number matrix
        array)  :Idler number matrix
    """
    N = len(K)
    Kss = K[0 : N // 2, 0 : N // 2]
    Ksi = K[0 : N // 2, N // 2 : N]
    Kiss = K[N // 2 : N, 0 : N // 2]
    # Constructing the moment matrix
    M = Kss @ (np.conj(Kiss).T)
    # Using SVD of M to construct JSA
    L, s, Vh = np.linalg.svd(M)
    Sig = np.diag(s)
    D = np.arcsinh(2 * Sig) / 2
    J = np.abs(L @ D @ Vh) / dk
    # Number of signal photons
    Nums = np.conj(Ksi) @ Ksi.T
    Numi = Kiss @ (np.conj(Kiss).T)
    Ns = np.real(np.trace(Nums))
    # Finding K    
    Schmidt = (np.trace(np.sinh(D) ** 2)) ** 2 / np.trace(np.sinh(D) ** 4)

    return J, Ns, Schmidt, M, Nums, Numi