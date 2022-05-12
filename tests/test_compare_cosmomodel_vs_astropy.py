import itertools
import typing
import unittest
import os
import sys
import numpy as np

import astropy.cosmology as cosmo
import astropy.units as u

from source.mcmc import MCMC

basedir = os.path.dirname(os.path.abspath(''))
sourcedir = os.path.join(basedir, 'source')
sys.path.insert(0, sourcedir)
datafile = os.path.join(basedir, "data/lcparam_DS17f.txt")
sysfile = os.path.join(basedir, 'data/sys_DS17f.txt')

class CosmoModelVsAstropy(unittest.TestCase):
    def setUp(self):
        from source.cosmo_model import CosmoModel
        self.CosmoModel = CosmoModel

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

        start = np.array([0.27, 0.73, 75, -19.2])
        markov_chain = MCMC(initial_state=start,
                            data_file=datafile,
                            systematics_file=sysfile,
                            g_cov=np.diag([.01, .01, .1, .01]))

        H0_range = np.arange(48, 108, 20)
        Om0_range = np.arange(0.25, 1, 0.25)
        Ode0_range = np.arange(0.25, 1, 0.25)
        M_range = np.arange(-25, -15, 5)

        total_len = len(H0_range)*len(Om0_range)*len(Ode0_range)*len(M_range)*len(z)  # *len(z_range)*len(mb_range)*len(M_range)

        for H0, Om0, Ode0, M, in itertools.product(H0_range, Om0_range, Ode0_range, M_range):
            # astropy_model = cosmo.LambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0, Ode0=Ode0)
            # astropy_distmod = np.array(astropy_model.distmod(z))
            # astropy_delta_mu = (mb - M) - astropy_distmod
            #
            # our_model = self.CosmoModel(Om0, Ode0, H0)
            # our_model_distmod = our_model.distmod(z)
            # our_model_delta_mu = (mb - M) - our_model_distmod

            params = np.array([H0, Om0, Ode0, M])
            our_distmod, our_dmu, our_chi2 = markov_chain.log_likelihood(params=params,
                                                                         astropy=False, return_value='all')
            ast_distmod, ast_dmu, ast_chi2 = markov_chain.log_likelihood(params=params,
                                                                         astropy=True, return_value='all')
            #distmod_diff = abs(our_distmod - ast_distmod)
            # print(f"Max difference between distmods: {max(distmod_diff)}")
            #
            # dmu_diff = abs(our_dmu - ast_dmu)
            # print(f"Max difference between dmu: {max(dmu_diff)}")

            # difference = abs(our_model_distmod - astropy_distmod)
            # difference = abs(our_model_delta_mu - astropy_delta_mu)
            difference = abs(our_chi2 - ast_chi2)
            percent_difference = round(abs(2*difference/(our_chi2 + ast_chi2)), 6)
            with self.subTest(params=params):
                # print(f"Astropy: {ast_chi2} | Our model: {our_chi2} | P. diff: {percent_difference}")
                self.assertLessEqual(percent_difference, 1e-4, f"{percent_difference} not <= 1e-5 ({params})")
            differences = np.append(differences, difference)
            # print(f"{z[:5]=}")
            # print(f"{H0=}, {Om0=}, {Ode0=}, {our_model_distmod=}, {astropy_distmod=}, {difference=}")

        successes += 1
        if successes % 100 == 0:
            # print(f"{successes=}/{total_len}")
            pass

        # print(f"Average difference: {np.average(differences)}")
        # print(f"Max difference: {np.max(differences)}")

        # num_pass = 0
        # for i in differences:
        #     if abs(i) < 1e-4:
        #         num_pass += 1
        # # print(f"Number passed: {num_pass}/{total_len}")
        # self.assertEqual(total_len, num_pass)

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

        my_model = self.CosmoModel(Om0, Ode0, H0)
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
