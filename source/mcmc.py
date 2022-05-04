import os
import numpy as np
from CosmoModel import CosmoModel

class MCMC:

    def __init__(self, initial_state, data_file, systematics_file=None):
        
        self._chain= [initial_state]
        self._initial_state = initial_state
        self._current_state = initial_state
        self._current_step = 0

        self._data_file = data_file

        if(systematics_file is not None):
            self._use_sys_cov = True
            self._systematics_file = systematics_file
        else:
            self._use_sys_cov = False

        binned_data = np.genfromtxt(data_file, usecols=(1,4,5))
        self._zcmb = binned_data[:,0]
        self._mb = binned_data[:, 1]
        self._dmb = binned_data[:, 2]

        self._cov = self.construct_covariance()
        self._fisher = np.linalg.pinv(self._cov) 
        #Fisher matrix is inverse of covariance matrix. Just inverting ahead of time.

    def construct_covariance(self):
        cov = np.diag(self._dmb)
        if(self._use_sys_cov):
            binned_sys = np.loadtxt(self._systematics_file)
            n = int(binned_sys[0])
            cov += binned_sys[1:].reshape((n,n))
        return cov

    #computes log_likelihood up to a constant (i.e. unnormalized)
    def log_likelihood(self, params):
        #params[0] = Omega_m, params[1]=Omega_L, params[2]=H0 [km/s/Mpc]
        cosmo = CosmoModel(params[0], params[1], params[2])
        mu_vector = cosmo.distmod(self._zcmb) - self._mb
        chi2 = np.einsum("i,ij,j", mu_vector.T, self._fisher, mu_vector)
        return -chi2/2.
