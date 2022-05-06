import os
from turtle import forward
import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import astropy.cosmology as cosmo
import astropy.units as u

from CosmoModel import CosmoModel


class MCMC:
    def __init__(self,
                 initial_state,
                 data_file: str,
                 systematics_file=None,
                 g_cov=np.diag([0.01, 0.01, .1, .1])) -> None: 

        self._chain = np.array(initial_state)
        self._initial_state = initial_state  # (Omega_m, Omega_L, H0, M)
        self._current_state = initial_state
        self._current_step = 0
        self._generating_cov = g_cov
        self._generating_fisher = np.linalg.pinv(g_cov)
        self._generating_det = np.linalg.det(g_cov)

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
        cov = np.diag(self._dmb)**2
        if self._use_sys_cov:
            binned_sys = np.loadtxt(self._systematics_file)
            n = int(binned_sys[0])
            cov += binned_sys[1:].reshape((n, n))
        return cov

    # computes log_likelihood up to a constant (i.e. unnormalized)
    def log_likelihood(self, params) -> float:
        """
        Takes in a vector of the parameters Omega_m, Omega_L, and H0, then creates a cosmological model
        off them and calculates the difference between
        :param params: Tuple of current parameters (Omega_m, Omega_, H0)
        :return: Numpy array of likelihood
        """
        # params[0] = Omega_m, params[1] = Omega_L, params[2] = H0 [km/s/Mpc]

        model = cosmo.LambdaCDM(H0=params[2]*u.km/u.s/u.Mpc, Om0 = params[0], Ode0=params[1])

        #cosmo = CosmoModel(params[0], params[1], params[2])
        mu_vector =  self._mb - model.distmod(self._zcmb).value() - params[3]  # difference of model_prediction - our_data

        # IDE thinks einsum can only return an array, but this returns a float, so next line ignores the warning
        # noinspection PyTypeChecker
        chi2: float = np.einsum("i,ij,j", mu_vector.T, self._fisher, mu_vector)
        return -chi2 / 2.

    def log_flat_priors(self, params):
        Om = params[0]
        Ol = params[1]
        H0 = params[2]
        M = params[3]
         
        log_p = 1.0

        #if(H0<50 or H0>100):
        #    log_p += -np.inf
        #elif(Om<0 or Om>1):
        #    log_p += -np.inf
        #elif(Ol<0 or Ol>1):
        #    log_p += -np.inf
        #elif(M<-25 or M>-15):
        #    log_p += -np.inf

        if(H0<50 or H0>100):
            log_p *= 0
        elif(Om<0 or Om>1):
            log_p *= 0
        elif(Ol<0 or Ol>1):
            log_p *= 0
        elif(M<-25 or M>-15):
            log_p *= 0

        return log_p

    def generator(self):
        """
        Generates a new candidate position for the chain
        :param position_in: Starting position list with [Omega_m, Omega_L, H0]
        :return: Candidate next position
        """
        new = np.random.multivariate_normal(mean=self._current_state, cov=self._generating_cov)
        while(new[0]<0 or new[1]<0):
            new = np.random.multivariate_normal(mean=self._current_state, cov=self._generating_cov)
        return new

    # equivalent to g(x,x')
    def move_probability(self, current_state, new_state):
        diff_vec = np.array(new_state) - np.array(current_state)
        norm = 1/(np.sqrt((2*np.pi)**3)*np.sqrt(self._generating_det))
        exponent = -0.5*np.einsum("i, ij, j", diff_vec.T, self._generating_fisher, diff_vec)
        return norm*np.exp(exponent)


    def generate_acceptance_prob(self,
                                 current_state,
                                 candidate_state) -> float:
        """
        Generates acceptance probability according to Metropolis-Hastings Algorithm.
        For right now, no priors are included.
        :param last_pos: Starting position
        :param candidate_pos: Potential new position
        :return: A probability for accepting the new position
        """
        new_log_likelihood = self.log_likelihood(candidate_state)
        back_prob = self.move_probability(candidate_state, current_state)
        forward_prob = self.move_probability(current_state, candidate_state)
        #back_prob = 0.0
        #forward_prob = 0.0
        #diff = new_log_likelihood + self.log_flat_priors(candidate_state) + back_prob - self._current_log_likelihood -self.log_flat_priors(current_state) - forward_prob

        diff = new_log_likelihood  + back_prob - self._current_log_likelihood - forward_prob
        
        return self.log_flat_priors(candidate_state)*np.exp(np.min([0, diff])), new_log_likelihood

    def propagate_chain(self) -> None:
        """
        Advances the Markov chain by one step
        :return: The propagated Markov chain
        """
        candidate_state = self.generator()  # new position x'
        acceptance_prob, new_log_likelihood = self.generate_acceptance_prob(self._current_state, candidate_state)

        random_number = np.random.uniform(0, 1)

        if random_number <= acceptance_prob or np.isnan(acceptance_prob):
            self._chain = np.vstack([self._chain, candidate_state])
            self._current_state = candidate_state
            self._current_log_likelihood = new_log_likelihood
        else:
            self._chain = np.vstack([self._chain, self._current_state])

        self._current_step += 1

    def make_chain(self, n):
        for _ in range(n):
            self.propagate_chain()
            
    @property
    def chain(self):
        return self._chain