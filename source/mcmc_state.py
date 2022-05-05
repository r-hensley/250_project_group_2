from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np

from mcmc import MCMC


@dataclass
class State:
    def __init__(self,
                 mcmc: MCMC,
                 *args: Tuple[Union[list, tuple, np.ndarray, float, ...]]) -> None:
        """
        A class containing information about a state in the MCMC algorithm.
        :param mcmc: The mcmc class instance to use the log_likelihood() function
        :param args: Takes either a list/array/tuple containing four parameters,
        or the four parameters as separate arguments
        """
        self._mcmc = mcmc
        if len(args) == 1:
            if type(args[0]) not in [list, tuple, np.ndarray]:
                raise TypeError(f"The single argument passed must be of type list, tuple, or np.ndarray. "
                                f"(type {type(args[0])} passed)")

            assert not [True for arg in args[0] if type(arg) != float]  # All elements of arg must be floats

            self._Omega_m = args[0][0]
            self._Omega_L = args[0][1]
            self._H0 = args[0][2]
            self._M = args[0][3]
        
        elif len(args) == 4:
            assert not [True for arg in args if type(arg) != float]  # All elements of arg must be floats

            self._Omega_m = args[0]
            self._Omega_L = args[1]
            self._H0 = args[2]
            self._M = args[3]

        else:
            raise TypeError(f"Please pass either one list/tuple/array argument or four separate arguments. "
                            f"({len(args)} arguments passed)")
        
    @property
    def tuple(self) -> tuple:
        """
        :return: A tuple containing the parameters (Omega_m, Omega_L, H0, M)
        """
        return self._Omega_m, self._Omega_L, self._H0, self._M

    @property
    def list(self) -> list:
        """
        :return: A list containing the parameters [Omega_m, Omega_L, H0, M]
        """
        return [self._Omega_m, self._Omega_L, self._H0, self._M]

    @property
    def array(self) -> np.ndarray:
        """
        :return: An array containing the parameters [Omega_m, Omega_L, H0, M]
        """
        return np.ndarray([self._Omega_m, self._Omega_L, self._H0, self._M])
        
    @property
    def likelihood(self) -> float:
        """
        Calculates and returns the log likelihood based on the log_likelihood() function in the MCMC class.
        :return: Log likelihood of the current state
        """
        return self._mcmc.log_likelihood(self)
