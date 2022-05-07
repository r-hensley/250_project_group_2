from dataclasses import dataclass
from typing import Union, Tuple, Any

import numpy as np


def is_float_or_int(number: Any) -> bool:
    """
    Checks whether a number is a float or an instance, including instances of numpy types such as float64
    :param number: Number to check
    :return: Boolean
    """
    return isinstance(number, (float, int, np.integer))

# Reference for subclassing np.ndarray:
# https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray

@dataclass
class State(np.ndarray):
    """
    A class containing information about a state in the MCMC algorithm.
    """

    def __new__(cls, *args):
        """
        The class takes input as a single array-like object or a series of four arguments
        :param args: Either a single array or exactly four float arguments
        """
        obj = super().__new__(cls, shape=(4,), dtype=float, buffer=None, offset=0, strides=None, order=None)
        print(f"in __new__ with cls {cls}")
        print(f"args in __new__: {args}")

        if len(args) == 1:
            if type(args[0]) not in [list, tuple, np.ndarray]:
                raise TypeError(f"The single argument passed must be of type list, tuple, or np.ndarray. "
                                f"(type {type(args[0])} passed)")

            assert not [True for arg in args[0] if not is_float_or_int(arg)], "Arguments must be ints or floats"

            obj._Omega_m = float(args[0][0])
            obj._Omega_L = float(args[0][1])
            obj._H0 = float(args[0][2])
            obj._M = float(args[0][3])

        elif len(args) == 4:
            assert not [True for arg in args if not is_float_or_int(arg)], "Arguments must be ints or floats"

            obj._Omega_m = float(args[0])
            obj._Omega_L = float(args[1])
            obj._H0 = float(args[2])
            obj._M = float(args[3])

        else:
            raise TypeError(f"Please pass either one list/tuple/array argument or four separate arguments. "
                            f"({len(args)} arguments passed)")

        print("end of __new__", obj, obj._Omega_m)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._Omega_m = getattr(obj, '_Omega_m', None)
        self._Omega_L = getattr(obj, '_Omega_L', None)
        self._H0 = getattr(obj, '_H0', None)
        self._M = getattr(obj, '_M', None)
        print(f"obj in __new__: {obj}")
        print("end of __array_finalize__", self._Omega_m)

    # def __init__(self, *args):
    #     super().__init__(self)

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
        return np.array([self._Omega_m, self._Omega_L, self._H0, self._M])

    def __iter__(self) -> np.ndarray:
        """
        Makes this class iterable to be accepted as an element of a numpy array
        :return: Numpy ndarray
        """
        return self.array

    def __str__(self) -> str:
        """
        Changes how this class is represented when printed as a string
        :return: String representation of object
        """
        return f"(Omega_m: {self._Omega_m}, " \
               f"Omega_L: {self._Omega_L}, " \
               f"H0: {self._H0}, " \
               f"M: {self._M})"

    def __repr__(self) -> str:
        """
        Changes how the class is represented when printed in development settings, or as a list element
        :return: Representation of string
        """
        params = [("Omega_m", round(self._Omega_m, 2)),
                  ("Omega_L", round(self._Omega_L, 2)),
                  ("H0", round(self._H0, 2)),
                  ("M", round(self._M, 2))]
        inner = ', '.join([f"{i[0]}: {i[1]}" for i in params])
        return f"<{self.__class__.__name__} {inner}>"

    def __getitem__(self, item: int) -> float:
        """
        Makes the class subscriptable
        :param item: Integer representing one of the elements of the class
        :return: Float representing one of the parameters
        """
        return self.list[item]


s1 = State(1,2,3,4)
s2 = State(5,6,7,8)

print(s1.tuple)
print(s1[0])

print(s1 + s2)
print(s1 * 4)

