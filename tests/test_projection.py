"""Fresh array-only preparation checks independent of source files and I/O."""

import unittest

import numpy as np
from scipy.interpolate import BSpline

from bayesiancalibration.data import project_loading_curves


class LoadingProjectionTest(unittest.TestCase):
    def test_fresh_arrays_match_independent_least_squares_and_qr(self):
        h_s = np.linspace(0.0, 20.0, 101)
        h_f = np.linspace(0.0, 18.0, 81)
        y_s = np.stack((h_s + .05 * h_s**2, 2 * h_s + .05 * h_s**2))
        y_f = np.stack((1.5 * h_f + .05 * h_f**2,
                        1.5 * h_f + .05 * h_f**2 + .1 * np.sin(h_f)))
        x = project_loading_curves(h_s, y_s, h_f, y_f)
        knots = np.r_[np.zeros(4), 1/3, 2/3, np.ones(4)]
        Phi_s = BSpline.design_matrix(x.library_t, knots, 3).toarray()[:, 1:]
        Phi_f = BSpline.design_matrix(x.field_t, knots, 3).toarray()[:, 1:]
        expected_c = np.linalg.lstsq(Phi_s, x.library_load_shifted.T, rcond=None)[0].T
        np.testing.assert_allclose(x.F_s, expected_c, rtol=2e-13, atol=2e-13)
        Q, R = np.linalg.qr(Phi_f, mode="reduced")
        signs = np.where(np.diag(R) < 0, -1., 1.)
        Q, R = Q * signs, signs[:, None] * R
        np.testing.assert_allclose(x.y_tilde, (Q.T @ y_f.T).T.reshape(-1), atol=1e-12)
        np.testing.assert_allclose(x.R, np.kron(np.eye(2), R), atol=1e-12)
        np.testing.assert_allclose(x.R.T @ x.R, np.kron(np.eye(2), Phi_f.T @ Phi_f),
                                   atol=2e-13)
        self.assertEqual(x.F_s.shape, (2, 5))
        self.assertEqual(x.y_tilde.shape, (10,))
        self.assertTrue(np.all(np.diag(x.R) > 0))

    def test_shared_grid_and_rank_contract(self):
        h = np.linspace(0.0, 20.0, 101)
        y = h[None, :]
        for invalid in (h[::-1], h[:, None], np.full_like(h, np.nan)):
            with self.subTest(depth=invalid.shape), self.assertRaisesRegex(ValueError, "sorted"):
                project_loading_curves(invalid, y, h, y)
        sparse_h = np.array([0., 4., 8., 12.])
        with self.assertRaisesRegex(ValueError, "full column rank"):
            project_loading_curves(sparse_h, sparse_h[None, :],
                                   sparse_h, sparse_h[None, :])


if __name__ == "__main__":
    unittest.main()
