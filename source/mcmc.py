from typing import Union
import os
import numpy as np

import astropy.cosmology as astropy_cosmo
import astropy.units as u

basedir = os.path.dirname(os.path.abspath(''))
sourcedir = os.path.join(basedir, 'source')
import sys
sys.path.insert(0, sourcedir)

from cosmo_model import CosmoModel


class MCMC:
    def __init__(self,
                 initial_state: np.ndarray,
                 data_file: str,
                 systematics_file=None,
                 g_cov=np.diag([0.01, 0.01, .1, .1]),
                 M_gaussian_prior=False) -> None:

        self._chain = np.array([initial_state], dtype=float)
        self._initial_state = initial_state  # (Omega_m, Omega_L, H0, M)
        self._current_state = initial_state
        self._current_step = 0  # maybe we don't need this?
        self._generating_cov = g_cov
        self._generating_fisher = np.linalg.pinv(g_cov)
        self._generating_det = np.linalg.det(g_cov)

        self._M_gaussian_prior = M_gaussian_prior

        self._data_file = data_file

        if systematics_file is not None:
            self._use_sys_cov = True
            self._systematics_file = systematics_file
        else:
            self._use_sys_cov = False

        binned_data = np.genfromtxt(data_file, usecols=(1, 4, 5))
        self._zcmb: np.ndarray = binned_data[:, 0]
        self._mb: np.ndarray = binned_data[:, 1]
        self._dmb: np.ndarray = binned_data[:, 2]

        self._cov = self.construct_covariance()  # covariance of data
        self._fisher = np.linalg.pinv(self._cov)

        self._current_log_likelihood = self.log_likelihood(initial_state)
        # Fisher matrix is inverse of covariance matrix. Just inverting ahead of time.

    def construct_covariance(self) -> np.ndarray:
        """
        Creates a 40x40 covariance matrix with the diagonal first coming from the dmb column of the lcparam_DS17f.txt
        file, then adds to the entire matrix (not just the diagonal) the systematic contribution from the entire
        sys_DS17f.txt file
        """
        cov = np.diag(self._dmb) ** 2  # 40 points on the diagonal
        if self._use_sys_cov:
            binned_sys = np.loadtxt(self._systematics_file)
            n = int(binned_sys[0])  # first line of text file, n=40
            cov += binned_sys[1:].reshape((n, n))  # last 1,600 lines, reshape into 40x40 matrix, add to cov matrix
        return cov

    # computes log_likelihood up to a constant (i.e. unnormalized)
    def log_likelihood(self, params: np.ndarray, astropy=False, return_value="chi2") -> Union[float, tuple, np.ndarray]:
        """
        Takes in a vector of the parameters Omega_m, Omega_L, and H0, then creates a cosmological model
        off them and calculates the difference between
        :param params: A tuple of current parameters (Omega_m, Omega_L, H0, M)
        :param astropy: Set to True to use astropy module for calculations
        :param return_value: Set variable to be returned. Defaults to chi2, can be chi2, delta_mu, distmod,
        or 'all' to return a tuple of (distmod, delta_mu, chi2)
        :return: Numpy array of likelihood
        """
        # params[0] = Omega_m, params[1] = Omega_L, params[2] = H0 [km/s/Mpc], params[3] = M

        if astropy:
            astropy_model = astropy_cosmo.LambdaCDM(H0=params[2] * u.km / u.s / u.Mpc, Om0=params[0], Ode0=params[1])
            distmod: np.ndarray = np.array(astropy_model.distmod(self._zcmb))
            mu_vector: np.ndarray = (self._mb - params[3]) - distmod
        else:
            cosmo = CosmoModel(params[0], params[1], params[2])  # instance of our model
            distmod: np.ndarray = cosmo.distmod(self._zcmb)
            mu_vector: np.ndarray = (self._mb - params[3]) - distmod   # difference of our_data - model_prediction

        # IDE thinks einsum can only return an array, but this returns a float, so next line ignores the warning
        # noinspection PyTypeChecker
        chi2: float = np.einsum("i,ij,j", mu_vector.T, self._fisher, mu_vector)
        if return_value == 'chi2':
            return -chi2 / 2.
        elif return_value == 'delta_mu':
            return mu_vector
        elif return_value == 'distmod':
            return distmod
        elif return_value == 'all':
            return distmod, mu_vector, -chi2 / 2
        else:
            raise ValueError("Invalid input for return_value parameter, must be chi2, delta_mu, or distmod")


    def priors(self, params: np.ndarray) -> float:
        """
        Depending on the four input parameters (Omega_m, Omega_L, H0, M), outputs a log probability which
        is zero outside of set ranges
        :param params: Four arguments (Omega_m, Omega_L, H0, M)
        :return: A single float probability either 0 or 1
        """
        Om = params[0]
        Ol = params[1]
        H0 = params[2]
        M = params[3]

        prior = 1.0

        if H0 < 50 or H0 > 100:
            prior *= 0
        elif Om < 0 or Om > 1:
            prior *= 0
        elif Ol < 0 or Ol > 1:
            prior *= 0

        if(self._M_gaussian_prior):
            gp = lambda Mag : 1./(np.sqrt(2*np.pi*0.042**2))*np.exp(-0.5*(Mag + 19.23)**2/(0.042**2))
            prior *= gp(M)
        else:
            if M < -25 or M > -15:
                prior *= 0

        return prior

    def generator(self) -> np.ndarray:
        """
        Generates a new candidate position for the chain
        :return: Candidate next position
        """
        new = np.random.multivariate_normal(mean=self._current_state,
                                            cov=self._generating_cov)

        while new[0] < 0 or new[1] < 0:  # don't allow it to generate Omega_m less than 0 or Omega_L greater than 1
            new = np.random.multivariate_normal(mean=self._current_state,
                                                cov=self._generating_cov)

        return new

    # equivalent to g(x,x')
    def move_probability(self,
                         current_state: np.ndarray,
                         new_state: np.ndarray) -> float:
        diff_vec = new_state - current_state
        norm = 1 / (np.sqrt((2 * np.pi) ** 3) * np.sqrt(self._generating_det))
        exponent = -0.5 * np.einsum("i, ij, j", diff_vec.T, self._generating_fisher, diff_vec)
        return norm * np.exp(exponent)

    def generate_acceptance_prob(self,
                                 current_state: np.ndarray,
                                 candidate_state: np.ndarray) -> (float, float):
        """
        Generates acceptance probability according to Metropolis-Hastings Algorithm.
        For right now, no priors are included.
        :param current_state: Starting position
        :param candidate_state: Potential new position
        :return: A probability for accepting the new position
        """
        new_log_likelihood = self.log_likelihood(candidate_state)
        back_prob = self.move_probability(candidate_state, current_state)
        forward_prob = self.move_probability(current_state, candidate_state)

        diff = new_log_likelihood + back_prob - self._current_log_likelihood - forward_prob

        return self.priors(candidate_state) * np.exp(np.min([0, diff])), new_log_likelihood

    def propagate_chain(self) -> None:
        """
        Advances the Markov chain by one step
        :return: The propagated Markov chain
        """
        candidate_state = self.generator()  # new position x'
        acceptance_prob, new_log_likelihood = self.generate_acceptance_prob(self._current_state, candidate_state)

        random_number = np.random.uniform(0, 1)

        if random_number <= acceptance_prob or np.isnan(acceptance_prob):
            self._chain = np.vstack([self.chain, candidate_state])
            self._current_state = candidate_state
            self._current_log_likelihood = new_log_likelihood
        else:
            self._chain = np.vstack([self.chain, candidate_state])

        self._current_step += 1

    def make_chain(self, n):
        for _ in range(n):
            self.propagate_chain()

    @property
    def chain(self):
        """
        :return: The chain object from the mcmc class
        """
        return self._chain

    @property
    def Omega_m(self) -> np.ndarray:
        """
        Returns only the Omega_m values out of the current Markov chain
        :return: Numpy array with values of Omega_m
        """
        return self.__getitem__(0)

    @property
    def Omega_L(self) -> np.ndarray:
        """
        Returns only the Omega_L values out of the current Markov chain
        :return: Numpy array with values of Omega_L
        """
        return self.__getitem__(1)

    @property
    def H0(self) -> np.ndarray:
        """
        Returns only the H0 values out of the current Markov chain
        :return: Numpy array with values of H0
        """
        return self.__getitem__(2)

    @property
    def M(self) -> np.ndarray:
        """
        Returns only the M values out of the current Markov chain
        :return: Numpy array with values of M
        """
        return self.__getitem__(3)

    def __getitem__(self, item: int) -> np.ndarray:
        """
        Makes this class subscriptable, pulls out an array containing all the data from the chain
        of just one of the parameters
        :param item: Integer corresponding to parameter (0: Omega_m, 1: Omega_L, 2: H0, 3: M)
        :return: Numpy array containing data of just one parameter
        """
        return np.array([i[item] for i in self.chain])
