import unittest
from typing import Union

import numpy as np
import scipy.integrate

if __name__ == "__main__":
    print("starting tests")
    loader = unittest.TestLoader()
    start_dir = "../tests"
    suite = loader.discover(start_dir)

    if suite:
        runner = unittest.TextTestRunner()
        runner.run(suite)
    # except AttributeError as e:
    #     print(e)
    #     raise
    #     # # necessary when running from jupyter notebook
    #     # print(e)
    #     # runner = unittest.TextTestRunner()
    #     # runner.run(suite)
    #     # # unittest.main(argv=['first-arg-is-ignored'], exit=False)


class CosmoModel:
    """
    This class defines a basic cosmological model, and from that model computes
    distances and the distance modulus. In this context, a model is defined by
    the parameters Omega_m, Omega_Lambda, and H0, and the parameter Omega_k
    is defined implicitly by Omega_k = 1 - Omega_m - Omega_L.

    A new CosmoModel object should be created for each set of parameters sampled
    in the MCMC.
    """

    def __init__(self, Omega_m: float, Omega_L: float, H0: float) -> None:
        """
        Initializes a cosmological model given input cosmological density parameters.
        Omega_k is defined so that \sum Omega_i = 1. The Omegas are defined such that
        Omega_i = rho_i / rho_critical.

        In this model, the radiation energy density Omega_r is ignored
        :param Omega_m: matter density parameter
        :param Omega_L: Lambda (dark energy) density parameter
        :param H0: Present value of Hubble constant. Should be in units of km/s/Mpc.
        """
        self._Omega_m = Omega_m
        self._Omega_L = Omega_L
        self._H0 = H0
        self._Omega_k = 1 - Omega_m - Omega_L

    def comoving(self, z: float) -> float:
        """
        Returns the comoving distance to a given redshift for the cosmological model.
        This is used in calculations of the luminosity distance.
        :param z: Redshift (flaat)
        :return: Comoving distance (float)
        """

        def integrand(redshift: float) -> float:
            value = np.sqrt(self._Omega_m * (1 + redshift) ** 3 + self._Omega_k * (1 + redshift) ** 2 + self._Omega_L)
            return 1. / value

        return scipy.integrate.quad(integrand, 0, z)[0]

    def dL(self, z: float) -> float:
        """
        Returns the luminosity distance to a given redshift for the cosmological model.
        Returns distance in units of Mpc.
        :param z: Redshift (float)
        :return: Luminosity distance (float)
        """
        c = 9.716e-15  # speed of light in Mpc/s
        to_hz = 3.241e-20  # conversion from km/s/Mpc to 1/s
        lum_distance = c / (self._H0 * to_hz) * (1 + z)  # prefactor in units of Mpc
        if self._Omega_k < 0:
            lum_distance *= (1. / np.sqrt(np.abs(self._Omega_k))) \
                            * np.sin(np.sqrt(np.abs(self._Omega_k)) * self.comoving(z))
        elif self._Omega_k == 0:
            lum_distance *= self.comoving(z)
        elif self._Omega_k > 0:
            lum_distance *= (1. / np.sqrt(np.abs(self._Omega_k))) \
                            * np.sinh(np.sqrt(np.abs(self._Omega_k)) * self.comoving(z))

        return lum_distance

    def distmod(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Returns the distance modulus for a given redshift.
        Performs operation on either a single float or an entire numpy array.
        :param z: Redshift (float or numpy array)
        :return: Distance modulus (float or numpy array, depending on input)
        """
        if type(z) == int:
            z = float(z)

        if type(z) in [np.int_, np.float_]:
            z = np.float64(z)

        assert type(z) in [float, np.float_, np.ndarray], f"Input into distmod() function is of wrong time. " \
                                                          f"Got {type(z)}, expected float or np.ndarray"

        if type(z) in [float, np.float_]:
            return 5 * np.log10(self.dL(z) / 1e-5)

        elif type(z) == np.ndarray:
            results = np.zeros(len(z))
            for i, j in enumerate(z):
                results[i] = 5 * np.log10(self.dL(j) / 1e-5)
            return results

        else:
            raise TypeError(f"Argument of invalid type passed for z in distmod(). Requires type float or np.ndarray, "
                            f"got type {type(z)}")
