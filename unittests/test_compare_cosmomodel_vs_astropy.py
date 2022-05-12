import itertools
import typing
import unittest
import os
import sys
import numpy as np

import astropy.cosmology as cosmo
import astropy.units as u

from source.CosmoModel import CosmoModel

basedir = os.path.dirname(os.path.abspath(''))
sourcedir = os.path.join(basedir, 'source')
sys.path.insert(0, sourcedir)
datafile = os.path.join(basedir, "data/lcparam_DS17f.txt")


class CosmoModelVsAstropy(unittest.TestCase):
    def test_parameter_agreement(self) -> None:
        """
        Loops over many values of H0, Omega_m, Omega_L, and M, and compares calculated values of delta_mu
        between the AstroPy package and our CosmoModel file
        """
        successes = 0
        differences = np.array([])
        binned_data = np.genfromtxt(datafile, usecols=(1, 4, 5))
        z = binned_data[:, 0]
        mb: np.ndarray = binned_data[:, 1]

        H0_range = np.arange(48, 108, 20)
        Om0_range = np.arange(0.25, 1, 0.25)
        Ode0_range = np.arange(0.25, 1, 0.25)
        M_range = np.arange(-25, -15, 5)

        total_len = len(H0_range)*len(Om0_range)*len(Ode0_range)*len(M_range)*len(z)  # *len(z_range)*len(mb_range)*len(M_range)

        for H0, Om0, Ode0, M, in itertools.product(H0_range, Om0_range, Ode0_range, M_range):
            astropy_model = cosmo.LambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0, Ode0=Ode0)
            astropy_distmod = np.array(astropy_model.distmod(z))
            astropy_delta_mu = (mb - M) - astropy_distmod

            our_model = CosmoModel(Om0, Ode0, H0)
            our_model_distmod = our_model.distmod(z)
            our_model_delta_mu = (mb - M) - our_model_distmod

            difference = abs(our_model_delta_mu - astropy_delta_mu)
            # difference = abs(our_model_distmod - astropy_distmod)
            differences = np.append(differences, difference)
            # print(f"{z[:5]=}")
            # print(f"{H0=}, {Om0=}, {Ode0=}, {our_model_distmod=}, {astropy_distmod=}, {difference=}")

        successes += 1
        if successes % 100 == 0:
            # print(f"{successes=}/{total_len}")
            pass

        # print(f"Average difference: {np.average(differences)}")
        # print(f"Max difference: {np.max(differences)}")

        num_pass = 0
        for i in differences:
            if abs(i) < 1e-4:
                num_pass += 1
        # print(f"Number passed: {num_pass}/{total_len}")
        self.assertEqual(total_len, num_pass)

    def test_simple_parameter_agreement(self) -> None:
        """
        A simpler test than test_parameter_agreement(), this locks values of H0, Omega_m, Omega_L
        and then just compares distmod
        """
        binned_data = np.genfromtxt(datafile, usecols=(1, 4, 5))
        z = binned_data[:, 0]
        differences = np.array([])

        Om0 = 0.23
        Ode0 = 0.77
        H0 = 68.5

        my_model = CosmoModel(Om0, Ode0, H0)
        my_dmoduli = my_model.distmod(z)

        astropy_model = cosmo.LambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0, Ode0=Ode0)
        a_dmoduli = np.array(astropy_model.distmod(z))

        difference = a_dmoduli - my_dmoduli

        # print(f"{z[:5]=}")
        # print(f"{H0=}, {Om0=}, {Ode0=}, {my_dmoduli=}, {a_dmoduli=}, {difference=}")

        differences = np.append(differences, difference)

        total_len = len(z)
        num_pass = 0
        for i in differences:
            if abs(i) < 1e-4:
                num_pass += 1
        # print(f"Number passed: {num_pass}/{total_len}")
        self.assertEqual(num_pass, total_len)


if __name__ == '__main__':
    unittest.main()
