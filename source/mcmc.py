import os

import numpy as np
from typing import Optional, List, Tuple

from matplotlib import pyplot as plt

from CosmoModel import CosmoModel


class MCMC:
    def __init__(self,
                 initial_state: Tuple[float, float, float],
                 data_file: str,
                 systematics_file: Optional[str],
                 g_sigma=(0.5, 0.5, 10)) -> None:

        self._chain = np.array([initial_state] + [(0., 0., 0.)] * 999)
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

        self._cov = self.construct_covariance()  # covariance of data
        self._fisher = np.linalg.pinv(self._cov)
        # Fisher matrix is inverse of covariance matrix. Just inverting ahead of time.

        self._parameter_space = [np.around(np.linspace(0.1, 1, 10), decimals=1),  # Omega_m
                                 np.around(np.linspace(0.1, 1, 10), decimals=1),  # Omega_L
                                 np.around(np.linspace(41, 70, 30), decimals=0),  # H0
                                 ]

        # A dictionary linking positions in parameter space to likelihoods
        # self._likelihood_lookup: Dict[Tuple[float, float, float]: float] = self.generate_likelihood_arrays()
        self.likelihoods = np.array([self.log_likelihood(initial_state)] + [0]*999)

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
        :param params: Tuple of current parameters (Omega_m, Omega_, H0)
        :return: Numpy array of likelihood
        """
        # params[0] = Omega_m, params[1] = Omega_L, params[2] = H0 [km/s/Mpc]
        cosmo = CosmoModel(params[0], params[1], params[2])
        mu_vector = cosmo.distmod(self._zcmb) - self._mb  # difference of model_prediction - our_data
        # IDE thinks einsum can only return an array, but this returns a float, so next line ignores the warning
        # noinspection PyTypeChecker
        chi2: float = np.einsum("i,ij,j", mu_vector.T, self._fisher, mu_vector)
        return -chi2 / 2.

    # def generate_likelihood_arrays(self) -> Dict[Tuple[float, float, float], float]:
    #     """
    #     Iterates over possible values for the model parameters and creates a likelihood distribution
    #     using the log_likelihood() function
    #     :return: A dictionary assigning values in parameter space to likelihood values
    #     """
    #     # self._parameter_space has arrays which contain possible values of the parameters
    #
    #     # Total number of points in parameter space
    #     total_parameter_num = np.prod([len(i) for i in self._parameter_space])
    #
    #     # An unused list of parameter values and likelihoods
    #     # params_to_likelihood: np.ndarray = np.array([((0, 0, 0), 0)]*1000)
    #
    #     # A dictionary of values in parameter space paired to likelihoods
    #     likelihood_lookup: Dict[Tuple[float, float, float]: float] = {}
    #
    #     params: Tuple[float, float, float]
    #     for idx, params in enumerate(itertools.product(*self._parameter_space)):
    #         if idx % 1000 == 0:
    #             print(f"Likelihood generation progress: {idx}/{total_parameter_num}")
    #         # Equivalent to a nested for loop over the three lists
    #         # i = an index that counts up from 0 over each loop
    #         # params = a tuple (Omega_m, Omega_L, H0)
    #         likelihood = self.log_likelihood(params)
    #         # params_to_likelihood[i] = (params, likelihood)
    #         likelihood_lookup[params] = likelihood
    #
    #     return likelihood_lookup

    def generator(self, position_in: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Generates a new candidate position for the chain
        :param position_in: Starting position list with [Omega_m, Omega_L, H0]
        :return: Candidate next position
        """
        candidate: List[float] = [0., 0., 0.]
        for i in range(3):
            candidate[i]: float = np.random.normal(
                loc=position_in[i],  # sample from gaussian centered around current pos.
                scale=self._generating_sigma[i])  # std. dev of distribution
            if i == 2:
                candidate[i] = round(candidate[i], 0)  # H0
            else:
                candidate[i] = round(candidate[i], 1)  # Omega_m, Omega_L
        return candidate[0], candidate[1], candidate[2]

    def generate_acceptance_prob(self,
                                 last_pos: Tuple[float, float, float],
                                 candidate_pos: Tuple[float, float, float]) -> float:
        """
        Generates acceptance probability according to Metropolis-Hastings Algorithm
        :param last_pos: Starting position
        :param candidate_pos: Potential new position
        :return: A probability for accepting the new position
        """
        # Posterior probability of last position P(x_t)
        last_pos_prob = self.likelihoods[self._current_step-1]
        candidate_pos_prob = self.log_likelihood(candidate_pos)

        # Posterior probability of new position P(x')
        # if candidate_pos in self._likelihood_lookup:
        #     candidate_pos_prob = self._likelihood_lookup[candidate_pos]
        # else:
        #     candidate_pos_prob = 0.

        # The probabilities to move back and forth from the new and old positions
        # g(x'|x_t)
        move_forward_prob = evaluate_gaussian(candidate_pos, last_pos, self._generating_sigma)
        # g(x_t|x')
        move_backward_prob = evaluate_gaussian(last_pos, candidate_pos, self._generating_sigma)

        acceptance_prob: float = min(1., candidate_pos_prob * move_backward_prob / last_pos_prob / move_forward_prob)
        return acceptance_prob

    def propagate_chain(self) -> None:
        """
        Advances the Markov chain by one time step
        :return: The propagated Markov chain
        """
        self._current_step += 1
        last_pos = tuple(self._chain[self._current_step - 1])  # starting position x_t
        candidate_pos = self.generator(last_pos)  # new position x'
        # for i, j in enumerate(candidate_pos):
        #     if j not in self._parameter_space[i]:
        #         new_parameter = find_closest(self._parameter_space[i], candidate_pos[i])
        #         print(f"Relocated point {candidate_pos[i]} in array {i} to {new_parameter}")
        #         candidate_pos[i] = new_parameter
        # candidate_pos = (candidate_pos[0], candidate_pos[1], candidate_pos[2])

        acceptance_prob = self.generate_acceptance_prob(last_pos, candidate_pos)

        random_number = np.random.uniform(0, 1)

        if random_number <= acceptance_prob:
            self._chain[self._current_step] = candidate_pos
        else:
            self._chain[self._current_step] = last_pos

    def make_chain(self):
        for _ in range(len(self._chain)-1):
            self.propagate_chain()
            print(f"Step #{self._current_step}")

    @property
    def chain(self):
        return self._chain


def evaluate_gaussian(pos: Tuple[float, float, float],
                      loc: Tuple[float, float, float],
                      scale: Tuple[float, float, float]) -> float:
    """
    Evaluates the probability density for a Gaussian
    :param pos: Position on the Gaussian
    :param loc: Mean value of Gaussian
    :param scale: Width of gaussian
    :return: Probability density
    """
    # normalization is wrong but for now it's close enough
    gaussian = np.sqrt((2 * np.pi) ** 3 * np.prod(scale))  # normalization
    for i in range(3):
        gaussian *= np.exp(-(pos[i] - loc[i]) ** 2 / (2 * scale[i] ** 2))
    return gaussian


# def find_closest(array: np.ndarray, value: Union[int, float]) -> float:
#     """
#     A function to find the closest element in an array to a specified value
#     :param array: Numpy array
#     :param value: Input value
#     :return: The element in the array that was closest to the specified value
#     """
#     differences = np.abs(array - value)  # Differences between elements of array and value
#     idx = differences.argmin()  # Index of the element of the array with the smallest difference
#     return array[idx]  # The element with the smallest difference


def main():
    basedir = os.path.dirname(os.path.abspath(''))
    datadir = os.path.join(basedir, 'data')

    binned_data_file = os.path.join(datadir, 'lcparam_DS17f.txt')
    binned_sys_file = os.path.join(datadir, 'sys_DS17f.txt')

    # The for loop is to allow for the option of plotting multiple chains on the same chart
    # (It just kinda looks cool)
    for _ in range(1):
        print(f"Starting Markov Chain #{_ + 1}")
        start = (round(np.random.uniform(0.1, 2), 1),  # Omega_m
                 round(np.random.uniform(0.1, 2), 1),  # Omega_L
                 round(np.random.uniform(1, 100), 0))  # H0
        markov_chain = MCMC(initial_state=start,
                            data_file=binned_data_file,
                            systematics_file=binned_sys_file,
                            g_sigma=(0.1, 0.1, 10))

        markov_chain.make_chain()

        figs, axs = plt.subplots(1, 3)
        # plt.rcParams['figure.figsize'] = [20, 7]
        for i in range(3):
            axs[i].plot(markov_chain.chain[:, i])

        plt.show()


if __name__ == "__main__":
    main()

print("Done")
