import numpy as np
import scipy.integrate

class CosmoModel:

    def __init__(self, Omega_m, Omega_L, H0):
        self._Omega_m = Omega_m
        self._Omega_L = Omega_L
        self._H0 = H0
        self._Omega_k = 1 - Omega_m - Omega_L

    def dL(self, z):
        c = 9.716e-9 ## speed of light in pc/s
        toHz = 3.241e-20 ## conversion from km/s/Mpc to 1/s
        lum_distance = c/(self._H0*toHz)*(1+z) ## prefactor in units of pc
        if(self._Omega_k < 0):
            lum_distance *= (1./np.sqrt(np.abs(self._Omega_k)))*np.sinh(np.sqrt(np.abs(self._Omega_k))*self.comoving(z))
        elif(self._Omega_k == 0):
            lum_distance *= self.comoving(z)
        elif(self._Omega_k > 0):
            lum_distance *= (1./np.sqrt(np.abs(self._Omega_k)))*np.sin(np.sqrt(np.abs(self._Omega_k))*self.comoving(z))

        return lum_distance

    def comoving(self, z):
        def integrand(redshift):
            return 1./np.sqrt(self._Omega_m*(1+redshift)**3 + self._Omega_L + self._Omega_k*(1+redshift)**2)

        return scipy.integrate.quad(integrand, 0, z)[0]
