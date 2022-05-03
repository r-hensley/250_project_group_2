import numpy as np
import scipy.integrate

class CosmoModel:
    """
        This class defines a basic cosmological model, and from that model computes
        distances and the distance modulus. In this context, a model is defined by 
        the parameters Omega_m, Omega_Lambda, and H0, and the parameter Omega_k
        is defined implicitly by Omega_k = 1 - Omega_m - Omega_L.

        A new CosmoModel object should be created for each set of parameters sampled 
        in the MCMC. 
    """
        
    def __init__(self, Omega_m, Omega_L, H0):
        """
        Initializes a cosmological model given input cosmological density parameters. 
        Omega_k is defined so that \sum Omega_i = 1. The Omegas are defined such that
        Omega_i = rho_i / rho_critical.
        
        Inputs:
        Omega_m - matter density parameter
        Omega_L - Lambda (dark energy) density parameter
        H0 - Present value of Hubble constant. Should be in units of km/s/Mpc.
        """
        self._Omega_m = Omega_m
        self._Omega_L = Omega_L
        self._H0 = H0
        self._Omega_k = 1 - Omega_m - Omega_L

    def dL(self, z):
        """
        Returns the luminosity distance to a given redshift for the cosmological model.
        Returns distance in units of Mpc.
        
        Inputs:
        z - redshift
        """
        c = 9.716e-15 ## speed of light in Mpc/s
        to_hz = 3.241e-20 ## conversion from km/s/Mpc to 1/s
        lum_distance = c/(self._H0*to_hz)*(1+z) ## prefactor in units of Mpc
        if(self._Omega_k < 0):
            lum_distance *= (1./np.sqrt(np.abs(self._Omega_k)))*np.sinh(np.sqrt(np.abs(self._Omega_k))*self.comoving(z))
        elif(self._Omega_k == 0):
            lum_distance *= self.comoving(z)
        elif(self._Omega_k > 0):
            lum_distance *= (1./np.sqrt(np.abs(self._Omega_k)))*np.sin(np.sqrt(np.abs(self._Omega_k))*self.comoving(z))

        return lum_distance

    def comoving(self, z):
        """
        Returns the comoving distance to a given redshift for the cosmological model.
        This is used in calculations of the luminosity distance.
        
        Inputs:
        z - redshift
        """
        def integrand(redshift):
            return 1./np.sqrt(self._Omega_m*(1+redshift)**3 + self._Omega_L + self._Omega_k*(1+redshift)**2)

        return scipy.integrate.quad(integrand, 0, z)[0]

    def distmod(self, z):
        """
        Returns the distance modulus for a given redshift.
        
        Inputs:
        z - redshift
        """
        return 5*np.log10(self.dL(z)/1e-5)

