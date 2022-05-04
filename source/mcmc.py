import itertools
import os

import numpy as np
from typing import Tuple, Optional, Union, List, Dict

from matplotlib import pyplot as plt

from CosmoModel import CosmoModel


class MCMC:
    def __init__(self,
                 initial_state: Tuple[float, float, float],
                 data_file: str,
                 systematics_file: Optional[str],
                 g_sigma=(0.5, 0.5, 10)) -> None:

        self._chain = np.array([initial_state])
        self._initial_state = initial_state  # (Omega_m, Omega_L, H0)
        self._current_state = initial_state
        self._current_step = 0
        self._generating_sigma = g_sigma

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

        self._cov = self.construct_covariance()
        self._fisher = np.linalg.pinv(self._cov)
        # Fisher matrix is inverse of covariance matrix. Just inverting ahead of time.

    def construct_covariance(self) -> np.ndarray:
        cov = np.diag(self._dmb)
        if self._use_sys_cov:
            binned_sys = np.loadtxt(self._systematics_file)
            n = int(binned_sys[0])
            cov += binned_sys[1:].reshape((n, n))
        return cov

    # computes log_likelihood up to a constant (i.e. unnormalized)
    def log_likelihood(self, params: Tuple[float, float, float]) -> float:
        """
        Takes in a vector of the parameters Omega_m, Omega_L, and H0, then creates a cosmological model
        off them and calculates the difference between
        :param params: Tuple of (Omega_m, Omega_, H0)
        :return: Numpy array of likelihood
        """
        # params[0] = Omega_m, params[1] = Omega_L, params[2] = H0 [km/s/Mpc]
        cosmo = CosmoModel(params[0], params[1], params[2])
        mu_vector = cosmo.distmod(self._zcmb) - self._mb  # difference of model_prediction - our_data
        # IDE thinks einsum can only return an array, but this returns a float, so next line ignores the warning
        # noinspection PyTypeChecker
        chi2: float = np.einsum("i,ij,j", mu_vector.T, self._fisher, mu_vector)
        return -chi2 / 2.

    def generate_likelihood_arrays(self) -> (np.ndarray, Dict[tuple: float]):
        """
        Iterates over possible values for the model parameters and creates a likelihood distribution
        using the log_likelihood() function
        :return: A dictionary assigning values in parameter space to likelihood values
        """
        # Create three arrays to contain possible values the parameters
        Omega_m: np.ndarray = np.linspace(0.05, 2, 40)
        Omega_L: np.ndarray = np.linspace(0.05, 2, 40)
        H0: np.ndarray = np.linspace(1, 200, 399)

        # Total number of points in parameter space
        total_parameter_num = len(Omega_m)*len(Omega_m)*len(H0)

        # A list of values in parameter space paired to likelihoods
        params_to_likelihood: np.ndarray = np.zeros(total_parameter_num)
        likelihood_lookup: Dict[Tuple[float, float, float]: float] = {}

        for i, params in enumerate(itertools.product(Omega_m, Omega_L, H0)):
            # Equivalent to a nested for loop over the three lists
            # i = an index that counts up from 0 over each loop
            # params = a tuple (Omega_m, Omega_L, H0)
            likelihood = self.log_likelihood(params)
            params_to_likelihood[i] = (params, likelihood)
            likelihood_lookup[params] = likelihood

        return params_to_likelihood, likelihood_lookup

    @staticmethod
    def evaluate_gaussian(pos: float,
                          loc: float,
                          scale: float) -> float:
        """
        Evaluates the probability density for a Gaussian
        :param pos: Position on the Gaussian
        :param loc: Mean value of Gaussian
        :param scale: Width of gaussian
        :return: Probability density
        """
        return 1 / (np.sqrt(2 * np.pi) * scale) * np.exp(-(pos - loc) ** 2 / (2 * scale ** 2))

    def generator(self, position_in: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Generates a new candidate position for the chain
        :param position_in: Starting position tuple with (
        :return: Candidate next position
        """
        candidate: List[float] = [0., 0., 0.]
        for i in range(3):
            candidate[i]: float = np.random.normal(
                loc=position_in[i],  # sample from gaussian centered around current pos.
                scale=self._generating_sigma[i])  # std. dev of distribution
            candidate[i] = round(candidate[i], 2)
        return candidate[0], candidate[1], candidate[2]

    def generate_acceptance_prob(self,
                                 last_pos: Tuple[float, float, float],
                                 candidate_pos: Tuple[float, float, float],
                                 likelihood_lookup: Dict[Tuple[float, float, float]: float]) -> float:
        """
        Generates acceptance probability according to Metropolis-Hastings Algorithm
        :param last_pos: Starting position
        :param candidate_pos: Potential new position
        :return: A probability for accepting the new position
        """
        # Posterior probability of last position P(x_t)
        last_pos_prob = likelihood_lookup[last_pos]

        # Posterior probability of new position P(x')
        if candidate_pos in likelihood_lookup:
            candidate_pos_prob = likelihood_lookup[candidate_pos]
        else:
            candidate_pos_prob = 0.

        # The probabilities to move back and forth from the new and old positions
        # g(x'|x_t)
        move_to_candidate_prob = self.evaluate_gaussian(candidate_pos, last_pos, self._generating_sigma)
        # g(x_t|x')
        move_to_last_pos_prob = self.evaluate_gaussian(last_pos, candidate_pos, self._generating_sigma)

        acceptance_prob: float = min(1.,
                                     candidate_pos_prob * move_to_last_pos_prob / last_pos_prob / move_to_candidate_prob)
        return acceptance_prob

    def propagate_chain(self, chain) -> Union[np.ndarray, list]:
        """
        Advances the Markov chain by one time step
        :return: The propagated Markov chain
        """
        last_pos = chain[-1]  # starting position x_t
        candidate_pos = self.generator(last_pos)  # new position x'
        if candidate_pos < 0.01:
            candidate_pos = 0.01

        acceptance_prob = self.generate_acceptance_prob(last_pos, candidate_pos)

        random_number = np.random.uniform(0, 1)

        if random_number <= acceptance_prob:
            chain.append(candidate_pos)
        else:
            chain.append(last_pos)

        return chain

    def make_chain(self):
        for _ in range(1000):
            self.chain = self.propagate_chain(self.chain)

        plt.rcParams['figure.figsize'] = [20, 7]
        plt.plot(self.chain)


if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(''))
    datadir = os.path.join(basedir, 'data')

    binned_data_file = os.path.join(datadir, 'lcparam_DS17f.txt')
    binned_sys_file = os.path.join(datadir, 'sys_DS17f.txt')

    for _ in range(5):
        start = (round(np.random.uniform(0.01, 2), 2),  # Omega_m
                 round(np.random.uniform(0.01, 2), 2),  # Omega_L
                 round(np.random.uniform(1, 200), 2))  # H0
        markov_chain = MCMC(initial_state=start,
                            data_file=binned_data_file,
                            systematics_file=binned_sys_file,
                            g_sigma=(0.1, 0.1, 10))

        markov_chain.make_chain()

print("Done")
