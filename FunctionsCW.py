"""Functions to produce JSA and moments for QPM and CW pump in (k,t) space"""

import numpy as np
from scipy.linalg import expm


def Total_propCW(k,t,ws,wi,Np,l):
    """Generates the total Heisenberg propagator

    Args:
        k (array): array of momentum values
        ws (array): dispersion relation for signal photons (function of momenta)
        wi (array): dispersion relation for idler photon (function of momenta)
        Np (float): Numper of pump photons/laser normalization
        l (float): length of nonlinear interaction region
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
        S = 1j*(np.sqrt(Np)*l*dk/(2*np.pi))*np.sinc(l*(k+k[:,np.newaxis])/(2*np.pi)) #Extra factors of pi due to np.sinc definition
        Q = np.block([[Rs,S],[np.conjugate(S),Ri]])
        K = expm(Q*dt)@K

    return K