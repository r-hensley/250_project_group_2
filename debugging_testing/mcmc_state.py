import numpy as np


def is_float_or_int(number) -> bool:
    """
    Checks whether a number is a float or an instance, including instances of numpy types such as float64
    :param number: Number to check
    :return: Boolean
    """
    return isinstance(number, (float, int, np.integer))

# Reference for subclassing np.ndarray:
# https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray


class State(np.ndarray):
    """
    A class containing information about a state in the MCMC algorithm.
    """

    def __new__(cls, *args):
        """
        The class takes input as a single array-like object or a series of four arguments
        :param args: Either a single array or exactly four float arguments
        """
        if len(args) == 1:
            if type(args[0]) not in [list, tuple, np.ndarray]:
                raise TypeError(f"The single argument passed must be of type list, tuple, or np.ndarray. "
                                f"(type {type(args[0])} passed)")

            assert not [True for arg in args[0] if not is_float_or_int(arg)], "Arguments must be ints or floats"
            obj = np.asanyarray(args[0], dtype=float).view(cls)  # normal numpy array reclassed as State type

        elif len(args) == 4:
            assert not [True for arg in args if not is_float_or_int(arg)], "Arguments must be ints or floats"
            obj = np.asanyarray(args, dtype=float).view(cls)  # normal numpy array reclasssed as State type

        else:
            raise TypeError(f"Please pass either one list/tuple/array argument or four separate arguments. "
                            f"({len(args)} arguments passed)")
        
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def tuple(self) -> tuple:
        """
        :return: A tuple containing the parameters (Omega_m, Omega_L, H0, M)
        """
        return self[0], self[1], self[2], self[3]

    @property
    def list(self) -> list:
        """
        :return: A list containing the parameters [Omega_m, Omega_L, H0, M]
        """
        return [self[0], self[1], self[2], self[3]]

    @property
    def array(self) -> np.ndarray:
        """
        :return: An array containing the parameters [Omega_m, Omega_L, H0, M]
        """
        return np.array([self[0], self[1], self[2], self[3]])

    # def __getitem__(self, idx: int) -> float:
    #     """
    #     Returns the item in the list of parameters
    #     """
    #     return self[idx]
        
    def __str__(self) -> str:
        """
        Changes how this class is represented when printed as a string
        :return: String representation of object
        """
        return f"(Omega_m: {self[0]}, " \
               f"Omega_L: {self[1]}, " \
               f"H0: {self[2]}, " \
               f"M: {self[3]})"

    def __repr__(self) -> str:
        """
        Changes how the class is represented when printed in development settings, or as a list element
        :return: Representation of string
        """
        _params = [("Omega_m", round(self[0], 2)),
                  ("Omega_L", round(self[1], 2)),
                  ("H0", round(self[2], 2)),
                  ("M", round(self[3], 2))]
        inner = ', '.join([f"{i[0]}: {i[1]}" for i in _params])
        return f"<{self.__class__.__name__} {inner}>"

    # def __len__(self):
    #     return len(self) > 0
    #
    # def __add__(self, other):
    #     if len(self) == len(other):
    #         return [self[i] + other[i] for i, _ in enumerate(self)]
    #     else:
    #         return self + other
    #
    # def __iadd__(self, other):
    #     return
    #
    # def __mul__(self, other):
    #     return [i*other for i in self]

    @property
    def Omega_m(self):
        return self[0]

    @property
    def Omega_L(self):
        return self[1]

    @property
    def H0(self):
        return self[2]

    @property
    def M(self):
        return self[3]


# s1 = State(1, 2, 3, 4)
# s2 = State(5, 6, 7, 8)

# print('s1 tuple', s1.tuple)
# print('s1[0]', s1[0])
#
# print('str s1', s1)
# print('repr s1', [s1])
#
# print('s1 values', s1.Omega_m, s1.Omega_L, s1.H0, s1.M)
# print('s2 values', s2.Omega_m, s2.Omega_L, s2.H0, s2.M)
#
# s3 = s1 + s2
# print('s3 0', s3[0])
# print('add two lists', s1 + s2)
# print('s1*4', s1 * 4)
