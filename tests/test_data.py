"""Independent spline, alignment, and supplied-data checks for Stage 14a."""

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.linalg import qr
from scipy.stats import linregress

from bayesiancalibration import data


SOURCE = Path(
    "/Users/jaehoonkim/Library/CloudStorage/OneDrive-TheOhioStateUniversity/"
    "Chkrebtii, Oksana's files - Jaehoon research/Reference/"
    "syn_theta_spatial_4chains_5re_Spatial"
)


class SplineAndAlignmentTest(unittest.TestCase):
    def test_basis_matches_R_splines_bs_reference(self) -> None:
        # Evaluated independently with R 4.6.1 splines::bs at these seven t.
        t = np.array([0.0, 0.1, 1 / 3, 0.5, 2 / 3, 0.9, 1.0])
        reference = np.array([
            [0, 0, 0, 0, 0],
            [.54225, .11025, .0045, 0, 0],
            [.25, 7 / 12, 1 / 6, 0, 0],
            [.03125, .46875, .46875, .03125, 0],
            [0, 1 / 6, 7 / 12, .25, 0],
            [0, .0045, .11025, .54225, .343],
            [0, 0, 0, 0, 1],
        ])
        np.testing.assert_allclose(data.loading_spline_basis(t), reference, atol=1e-14)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            data.loading_spline_basis(np.array([-0.01, 0.5]))

    def test_local_ols_with_irregular_unsorted_repeated_per_curve_depths(self) -> None:
        library_depth = np.array([
            [7.0, 0.0, 3.0, 1.0, 1.0, 4.7, 5.2, 8.0, 2.0, 6.0],
            [0.2, 8.2, 4.4, 3.3, 1.1, 2.2, 6.5, 7.8, 5.6, 0.2],
        ])
        field_depth = np.array([
            [4.2, 0.0, 0.0, 0.9, 2.7, 3.8, 5.1, 7.0, 8.0, 9.0],
            [1.3, 0.4, 2.8, 4.2, 5.0, 6.0, 7.0, 8.0, 0.4, 3.7],
        ])
        library_load = 2.0 * library_depth + np.array([[3.0], [-7.0]])
        field_load = 2.0 * field_depth + np.array([[10.0], [20.0]])
        h0, candidates, errors, library_slopes, field_slopes = (
            data.match_loading_offset(library_depth, library_load,
                                      field_depth, field_load)
        )
        self.assertAlmostEqual(h0, 0.2)
        self.assertAlmostEqual(candidates[0], 0.2)
        np.testing.assert_allclose(np.diff(candidates), np.diff(candidates)[0])
        self.assertLessEqual(np.diff(candidates)[0], 0.8)
        np.testing.assert_allclose(field_slopes, 2.0)
        np.testing.assert_allclose(library_slopes[np.isfinite(library_slopes)], 2.0)
        np.testing.assert_allclose(errors[np.isfinite(errors)], 0.0, atol=1e-14)

    def test_window_slopes_match_independent_scipy_reference(self) -> None:
        h = np.arange(0.0, 20.0, 0.8)
        library = np.stack([0.5 * h**2, 0.5 * h**2 + 7.0])
        field = np.stack([(0.5 * h**2 + 2.2 * h)] * 3)
        h0, candidates, errors, library_slopes, field_slopes = (
            data.match_loading_offset(h, library, h, field)
        )
        field_window = h <= 4.0
        reference_field = linregress(h[field_window], field[0, field_window]).slope
        np.testing.assert_allclose(field_slopes, reference_field)
        for i in range(len(candidates)):
            inside = (h >= candidates[i]) & (h <= candidates[i] + 4.0)
            reference = np.mean([
                linregress(h[inside], row[inside]).slope for row in library
            ])
            self.assertAlmostEqual(library_slopes[i], reference, places=12)
            self.assertAlmostEqual(errors[i], abs(reference - reference_field),
                                   places=12)
        self.assertIn(h0, candidates)
        np.testing.assert_allclose(np.diff(candidates), np.diff(candidates)[0])
        self.assertLessEqual(errors[np.where(candidates == h0)[0][0]],
                             np.min(errors) + 1e-12)
        finer = data.match_loading_offset(h, library, h, field,
                                          candidate_step=0.4)[1]
        self.assertGreater(len(finer), len(candidates))
        self.assertLessEqual(np.diff(finer).max(), 0.4 + 1e-12)
        with self.assertRaisesRegex(ValueError, "candidate_step"):
            data.match_loading_offset(h, library, h, field,
                                      candidate_step=0.0)

    def test_degenerate_local_depth_variance_is_rejected(self) -> None:
        library_depth = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        field_depth = 1000.0 + np.arange(7) * 1e-12
        with self.assertRaisesRegex(ValueError, "variance"):
            data.match_loading_offset(library_depth, library_depth[None, :],
                                      field_depth, field_depth[None, :])


@unittest.skipUnless(
    all((SOURCE / name).is_file() for name in data._FILES)
    and find_spec("openpyxl") is not None
    and find_spec("matplotlib") is not None,
    "Supplied sources or optional preprocessing packages are unavailable",
)
class SuppliedSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = data.prepare_synthetic_data(SOURCE, seed=1024)

    def test_shapes_coordinates_library_scale_and_fixed_noise(self) -> None:
        x = self.prepared
        self.assertEqual(x.F_s.shape, (20, 5))
        self.assertEqual(x.y_tilde.shape, (300,))
        self.assertEqual(x.R.shape, (300, 300))
        np.testing.assert_array_equal(x.s[:20, 0], np.arange(0, 120, 6))
        np.testing.assert_array_equal(x.s[20:40, 1], 6)
        np.testing.assert_array_equal(x.s[40:, 1], 0)
        np.testing.assert_allclose(x.theta_bar_dagger, x.theta_s_dagger.mean(axis=0))
        np.testing.assert_allclose(np.diag(x.D_theta),
                                   x.theta_s_dagger.std(axis=0, ddof=1))
        np.testing.assert_allclose(x.theta_s_tilde.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(x.theta_s_tilde.std(axis=0, ddof=1), 1)
        self.assertEqual(x.metadata["field_noise"]["seed"], 1024)
        self.assertEqual(x.metadata["branch_sizes"], [5])
        self.assertEqual(x.metadata["load_scaling"], "unchanged numeric values")
        self.assertLess(x.metadata["offset_rule"]["h0"], 0.8)
        self.assertAlmostEqual(
            x.metadata["offset_rule"]["absolute_error_at_h0"],
            float(np.nanmin(x.slope_errors)), places=15,
        )

    def test_qr_projection_and_library_fit_against_independent_linear_algebra(self) -> None:
        x = self.prepared
        Phi_s = data.loading_spline_basis(x.library_t)
        Phi_f = data.loading_spline_basis(x.field_t)
        # The fitted library residual must be orthogonal to all five columns.
        residual = x.library_load_shifted.T - Phi_s @ x.F_s.T
        np.testing.assert_allclose(Phi_s.T @ residual, 0.0, atol=5e-7)
        q, _ = qr(Phi_f, mode="economic")
        np.testing.assert_allclose(
            np.sum(x.field_load_noisy**2, axis=1),
            np.sum(x.y_tilde.reshape(60, 5)**2, axis=1)
            + np.sum((x.field_load_noisy.T - q @ (q.T @ x.field_load_noisy.T))**2,
                     axis=0),
            rtol=1e-12,
        )
        np.testing.assert_allclose(x.R, np.kron(np.eye(60), x.R[:5, :5]))
        self.assertTrue(np.all(np.diag(x.R[:5, :5]) > 0))

    def test_truth_is_evaluation_only_and_archive_is_pickle_free(self) -> None:
        x = self.prepared
        original = data._load_csv

        def changed_truth(path, *, header=None):
            value = original(path, header=header)
            if path.name == "theta_spatial.csv":
                return value + 100.0
            return value

        with patch.object(data, "_load_csv", side_effect=changed_truth):
            changed = data.prepare_synthetic_data(SOURCE, seed=1024)
        np.testing.assert_array_equal(changed.F_s, x.F_s)
        np.testing.assert_array_equal(changed.theta_s_tilde, x.theta_s_tilde)
        np.testing.assert_array_equal(changed.y_tilde, x.y_tilde)
        np.testing.assert_array_equal(changed.R, x.R)
        np.testing.assert_array_equal(changed.theta_f_truth_for_evaluation,
                                      x.theta_f_truth_for_evaluation + 100)

        with tempfile.TemporaryDirectory() as temporary:
            data.save_prepared(x, temporary)
            self.assertTrue((Path(temporary) / "slope_error.png").is_file())
            with np.load(Path(temporary) / "synthetic_preprocessing.npz",
                         allow_pickle=False) as archive:
                self.assertEqual(archive["y_tilde"].shape, (300,))
                self.assertEqual(json.loads(archive["metadata"].item())["stage"],
                                 "14a")
                np.testing.assert_array_equal(archive["field_noise"], x.field_noise)


if __name__ == "__main__":
    unittest.main()
