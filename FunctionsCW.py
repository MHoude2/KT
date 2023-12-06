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
    
    
    for i in t:
        S = 1j*(np.sqrt(Np)*l*dk/(2*np.pi))*np.sinc(l*(k+k[:,np.newaxis])/(2*np.pi))*np.exp(1j*(ws+wi[:,np.newaxis])*i) #Extra factors of pi due to np.sinc definition
        Q = np.block([[0*np.eye(len(S)),S],[np.conjugate(S),0*np.eye(len(S))]])
        K = expm(Q*dt)@K

    return K