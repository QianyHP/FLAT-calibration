"""Numerical contracts for the reference-only IVW utility (no SUMO)."""
import unittest
import numpy as np
from compute_ivw_weights import inverse_variance_weights

class IVWTests(unittest.TestCase):
    def test_known_variances(self):
        w,v=inverse_variance_weights([[0,0],[1,2],[2,4]])
        np.testing.assert_allclose(v,[1,4])
        np.testing.assert_allclose(w,[0.8,0.2])

    def test_zero_variance_requires_explicit_policy(self):
        with self.assertRaises(ValueError): inverse_variance_weights([[0,0],[1,0]])
        w,_=inverse_variance_weights([[0,0],[1,0]],variance_floor=0.1)
        np.testing.assert_allclose(w,[1/6,5/6])

    def test_invalid_input(self):
        for x in [[[1,2]],[[1,2],[3,np.nan]]]:
            with self.assertRaises(ValueError): inverse_variance_weights(x)

    def test_common_rescaling_and_row_order(self):
        x=np.array([[0,1],[1,4],[4,9.]])
        a,_=inverse_variance_weights(x)
        b,_=inverse_variance_weights(7*x[::-1])
        np.testing.assert_allclose(a,b)

if __name__=='__main__': unittest.main()
