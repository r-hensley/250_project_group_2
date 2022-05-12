import unittest
import astropy.cosmology as cosmo
import astropy.units as u


class CosmoModelVsAstropy(unittest.TestCase):
    def test_something(self):
        # params[0] = Omega_m, params[1] = Omega_L, params[2] = H0 [km/s/Mpc], params[3] = M
        model = cosmo.LambdaCDM(H0=params[2] * u.km / u.s / u.Mpc, Om0=params[0], Ode0=params[1])
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
