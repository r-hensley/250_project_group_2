import os
import numpy as np

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

    def construct_covariance(self):
        cov = np.diag(self._dmb)
        if(self._use_sys_cov):
            binned_sys = np.loadtxt(self._systematics_file)
            n = int(binned_sys[0])
            cov += binned_sys[1:].reshape((n,n))
        return cov
        
