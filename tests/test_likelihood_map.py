import unittest
import os

import matplotlib.pyplot as plt
import numpy as np

from source.mcmc import MCMC



basedir = os.path.dirname(os.path.abspath(''))
sourcedir = os.path.join(basedir, 'source')
datadir = os.path.join(basedir, 'data')
binned_data_file = os.path.join(datadir, 'lcparam_DS17f.txt')
binned_sys_file = os.path.join(datadir, 'sys_DS17f.txt')


class LikelihoodMap(unittest.TestCase):
    def test_1d_likelihoods(self) -> None:
        """
        A test to make 1D likelihood maps of parameters assuming the others are all the correct
        theoretical values and seeing if they get peaks at the points of their correct values as well.
        """
        debug = False
        max_likelihoods = []

        def setup_chain():
            # % config
            # InlineBackend.figure_format = 'retina'
            # font = {'size': 16, 'family': 'STIXGeneral'}
            # axislabelfontsize = 'medium'
            # matplotlib.rc('font', **font)
            # matplotlib.mathtext.rcParams['legend.fontsize'] = 'medium'
            # plt.rcParams["figure.figsize"] = [8.0, 10.0]

            start = [0.27, 0.73, 73.8, -19.2]  # Omega_m, Omega_L, H0, M
            start = np.array(start)

            markov_chain = MCMC(initial_state=start,
                                data_file=binned_data_file,
                                systematics_file=binned_sys_file,
                                g_cov=np.diag([.01, .01, .1, .01]))

            return markov_chain, start

        markov_chain, start = setup_chain()

        def main():
            def map_likelihood(arg: int):
                """
                Map the argument in start parameter assuming the others
                :param arg: 0: Omega_m, 1: Omega_L, 2: H0, 3: M
                :return: Likelihood x and y arrays
                """
                if arg == 0 or arg == 1:
                    min_value = 0
                    max_value = 1
                    step = 0.05
                    if arg == 0:
                        variable = "Omega_m"
                    else:
                        variable = "Omega_L"
                elif arg == 2:
                    min_value = 50
                    max_value = 100
                    step = 1
                    variable = "H0"
                elif arg == 3:
                    min_value = -25
                    max_value = -15
                    step = 0.5
                    variable = "M"
                else:
                    raise ValueError

                likelihood_x = np.around(np.arange(min_value, max_value, step), decimals=2)
                likelihood_y = []
                astropy_likelihood_y = []

                if debug:
                    print()
                    print(f"Currently varying {variable}")

                for i in likelihood_x:
                    input = np.array([start[0], start[1], start[2], start[3]])
                    input[arg] = i
                    our_l = markov_chain.log_likelihood(input, astropy=False)
                    likelihood_y.append((input, our_l))

                    astro_l = markov_chain.log_likelihood(input, astropy=True)
                    astropy_likelihood_y.append((input, astro_l))

                    if debug:
                        print(f"{arg}: {input} --> Our: {round(our_l, 3)} | Astro: {round(astro_l, 3)}")

                return likelihood_x, likelihood_y, astropy_likelihood_y

            if debug:
                fig, ax = plt.subplots(4, 1)
                print("Shown below is the parameters input to log_likelihood() as well as the likelihood they give out.")
            for var in range(4):
                x, y, astro_y = map_likelihood(var)
                y = [i[1] for i in y]
                astro_y = [i[1] for i in astro_y]
                y = astro_y
                max_value = max(y)
                max_value_parameter = x[np.argmax(y)]
                if debug:
                    ax[var].plot(x, y)
                    # ax[i].plot(x, astro_y)
                    # ax[i].legend()
                    print("before max values")
                    print(var, max_value_parameter, max_value)
                max_likelihoods.append(max_value_parameter)

            if debug:
                plt.show()

            for var_num, max_likelihood in enumerate(max_likelihoods):
                if var_num == 0:  # Omega_m
                    with self.subTest():  # varying_parameter="Omega_m"
                        self.assertTrue(0.2 < max_likelihood <= 0.5,
                                        f"Max value of Omega_m in 1D likelihood map test ({max_value_parameter}) "
                                        f"not in range of 0.2 ~ 0.5.")
                        print(f"Parameter Omega_m has max at {max_likelihood} within range of 0.2 ~ 0.5.")
                if var_num == 1:  # Omega_L
                    with self.subTest(varying_parameter="Omega_L"):
                        self.assertTrue(0.5 <= max_likelihood < 0.9,
                                        f"Max value of Omega_L in 1D likelihood map test ({max_likelihood}) "
                                        f"not in range of 0.5 ~ 0.9.")
                        print(f"Parameter Omega_L properly has max at {max_likelihood} within range of 0.5 ~ 0.9.")
                if var_num == 2:  # H0
                    with self.subTest(varying_parameter="H0"):
                        self.assertTrue(60 < max_likelihood < 80,
                                        f"Max value of H0 in 1D likelihood map test ({max_likelihood}) "
                                        f"not in range of 60 ~ 80.")
                        print(f"Parameter H0 properly has max at {max_likelihood} within range of 60 ~ 80.")
                if var_num == 3:  # M
                    with self.subTest(varying_parameter="M"):
                        self.assertTrue(-22 < max_likelihood < -18,
                                        f"Max value of M in 1D likelihood map test ({max_likelihood}) "
                                        f"not in range of -22 ~ -18.")
                        print(f"Parameter M properly has max at {max_likelihood} within range of -22 ~ -18.")

        main()


if __name__ == '__main__':
    print("below in bottom", __name__)
    unittest.main()
