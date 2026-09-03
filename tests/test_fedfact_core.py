import unittest
import numpy as np
import torch

from algorithm.fedfact_core import (
    FedFACTConfig, SupportError, build_support_statistics,
    initialize_fedfact_state, project_nonnegative_l1_ball,
    validate_fedfact_state,
)
from tests.fedfact_test_utils import TinyTextDataset, make_datasets_and_loaders, seeded_model


class ConfigSupportStateTest(unittest.TestCase):
    def test_configuration_is_explicit_and_all_client(self):
        raw = {
            "task": "SENT_CLF", "num_clients_K": 2, "FL_fraction": 1.0,
            "FL_drop_rate": 0.0, "fairness_metric": "EO",
            "global_constraint": .1, "local_constraint": .2,
            "dual_learning_rate": .3, "dual_bound": 5.0, "dual_init": .1,
            "ensemble_learning_rate": .4, "ensemble_weight_init": .5,
            "calibration_epsilon": .001,
        }
        self.assertEqual(FedFACTConfig.from_param_dict(raw).num_constraints, 2)
        with self.assertRaisesRegex(ValueError, "FL_fraction == 1.0"):
            FedFACTConfig.from_param_dict(dict(raw, FL_fraction=.5))
        with self.assertRaisesRegex(ValueError, "FL_drop_rate == 0.0"):
            FedFACTConfig.from_param_dict(dict(raw, FL_drop_rate=.1))
        with self.assertRaisesRegex(ValueError, "DP or EO"):
            FedFACTConfig.from_param_dict(dict(raw, fairness_metric="DEO"))
        with self.assertRaisesRegex(ValueError, "SENT_CLF"):
            FedFACTConfig.from_param_dict(dict(raw, task="IMG_CLF"))

    def test_support_axes_are_client_protected_label(self):
        datasets, _ = make_datasets_and_loaders()
        stats = build_support_statistics(datasets, metric="EO")
        self.assertEqual(tuple(stats.counts.shape), (2, 2, 2))
        torch.testing.assert_close(stats.counts[0], torch.ones((2, 2), dtype=torch.float64))
        torch.testing.assert_close(stats.client_totals, torch.tensor([4., 4.], dtype=torch.float64))

    def test_missing_dp_group_and_eo_cell_fail_closed(self):
        no_group_one = TinyTextDataset([(-1., 0, 0), (1., 1, 0)])
        with self.assertRaisesRegex(SupportError, r"client 0.*protected=1"):
            build_support_statistics([no_group_one], metric="DP")
        no_a1_y1 = TinyTextDataset([(-2., 0, 0), (-1., 1, 0), (1., 0, 1)])
        with self.assertRaisesRegex(SupportError, r"client 0.*protected=1.*label=1"):
            build_support_statistics([no_a1_y1], metric="EO")

    def test_projection_is_over_the_whole_nonnegative_l1_set(self):
        actual = project_nonnegative_l1_ball(
            torch.tensor([.8, .6, -.2], dtype=torch.float64), bound=1.0
        )
        torch.testing.assert_close(actual, torch.tensor([.6, .4, 0.], dtype=torch.float64))
        self.assertLessEqual(actual.sum().item(), 1.0)

    def test_state_has_independent_cpu_personal_models_and_shapes(self):
        model = seeded_model()
        datasets, _ = make_datasets_and_loaders()
        stats = build_support_statistics(datasets, metric="EO")
        state = initialize_fedfact_state(
            model, 2, "EO", stats, dual_init=.1, ensemble_weight_init=.5
        )
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(state["variant"], "fedfact_in")
        self.assertEqual(tuple(state["global_dual"].shape), (2, 2))
        self.assertEqual(tuple(state["local_duals"].shape), (2, 2, 2))
        self.assertEqual(tuple(state["ensemble_weights"].shape), (2,))
        self.assertIsNot(state["personal_model_states"][0], state["personal_model_states"][1])
        self.assertTrue(all(
            tensor.device.type == "cpu"
            for client_state in state["personal_model_states"]
            for tensor in client_state.values()
        ))
        validate_fedfact_state(state, model, 2, "EO", stats, dual_bound=5.0)

from algorithm.fedfact_core import build_calibration_matrices, calibrated_loss


class CalibrationMatrixAndLossTest(unittest.TestCase):
    def setUp(self):
        # counts[k,a,y]; every global (a,y) count is four and every client group has four.
        self.counts = torch.tensor([
            [[3., 1.], [1., 3.]],
            [[1., 3.], [3., 1.]],
        ], dtype=torch.float64)

    def test_dp_matrix_matches_get_cal_matrix_probability_factors(self):
        matrices = build_calibration_matrices(
            client_id=0,
            support_counts=self.counts,
            metric="DP",
            global_dual=torch.tensor([[.4, .1]], dtype=torch.float64),
            local_dual=torch.tensor([[.2, .05]], dtype=torch.float64),
            epsilon=.001,
        )
        expected = torch.tensor([
            [[2.201, 2.401], [1.201, 3.401]],
            [[2.201, .001], [1.201, 1.001]],
        ], dtype=torch.float64)
        torch.testing.assert_close(matrices, expected, rtol=0, atol=1e-12)

    def test_eo_matrix_matches_both_label_conditioned_terms(self):
        # Constraint axis order is natural label order [y=0, y=1].
        matrices = build_calibration_matrices(
            client_id=0,
            support_counts=self.counts,
            metric="EO",
            global_dual=torch.tensor([[.2, 0.], [.1, 0.]], dtype=torch.float64),
            local_dual=torch.tensor([[.05, 0.], [.02, 0.]], dtype=torch.float64),
            epsilon=.001,
        )
        expected = torch.tensor([
            [[2.601, 2.6676666666666664], [1.601, 3.321]],
            [[2.601, .001], [1.601, 2.094333333333333]],
        ], dtype=torch.float64)
        torch.testing.assert_close(matrices, expected, rtol=0, atol=1e-12)

    def test_calibrated_loss_is_only_selected_matrix_row_times_log_softmax(self):
        matrix = torch.tensor([
            [[2.201, 2.401], [1.201, 3.401]],
            [[2.201, .001], [1.201, 1.001]],
        ], dtype=torch.float64)
        logits = torch.log(
            torch.tensor([[.8, .2], [.25, .75]], dtype=torch.float32)
        ).detach().requires_grad_(True)
        labels = torch.tensor([0, 1])
        protected = torch.tensor([1, 0])
        expected = -(
            2.201 * np.log(.8) + .001 * np.log(.2)
            + 1.201 * np.log(.25) + 3.401 * np.log(.75)
        ) / 2
        actual = calibrated_loss(logits, labels, protected, matrix)
        self.assertAlmostEqual(actual.item(), expected, places=6)

        actual.backward()
        self.assertIsNotNone(logits.grad)

from algorithm.fedfact_core import (
    ensemble_probabilities, update_ensemble_weight,
    confusion_from_predictions, disparity_from_confusion, update_dual,
)


class EnsembleAndDualTest(unittest.TestCase):
    def test_probability_mixture_has_a_different_decision_than_logit_mixture(self):
        theta = torch.tensor([[0., -10.]])
        phi = torch.tensor([[0., .5]])
        probability_mix = ensemble_probabilities(theta, phi, weight=.1)
        probability_prediction = probability_mix.argmax(1).item()
        logit_prediction = (.1 * theta + .9 * phi).argmax(1).item()
        self.assertEqual(probability_prediction, 1)
        self.assertEqual(logit_prediction, 0)

    def test_weight_moves_toward_the_lower_loss_unified_model(self):
        actual = update_ensemble_weight(
            weight=torch.tensor(.5, dtype=torch.float64),
            theta_loss=.2,
            phi_loss=.8,
            learning_rate=.3,
        )
        self.assertAlmostEqual(actual.item(), .5448788923735801, places=12)
        self.assertGreater(actual.item(), .5)

    def test_dp_and_eo_signed_disparities_use_positive_prediction_rates(self):
        predictions = torch.tensor([0, 1, 1, 1, 0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        protected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        confusion = confusion_from_predictions(predictions, labels, protected)
        self.assertEqual(tuple(confusion.shape), (2, 2, 2))
        torch.testing.assert_close(
            disparity_from_confusion(confusion, "DP"),
            torch.tensor([-.25], dtype=torch.float64),
        )
        torch.testing.assert_close(
            disparity_from_confusion(confusion, "EO"),
            torch.tensor([-.5, 0.], dtype=torch.float64),
        )

    def test_dual_has_distinct_positive_negative_residuals_and_l1_projection(self):
        current = torch.zeros((1, 2), dtype=torch.float64)
        positive = update_dual(current, torch.tensor([.3]), tolerance=.1,
                               learning_rate=.5, bound=1.)
        torch.testing.assert_close(positive, torch.tensor([[.1, 0.]], dtype=torch.float64))
        projected = update_dual(
            torch.tensor([[.8, .6]], dtype=torch.float64),
            torch.tensor([.1]), tolerance=0., learning_rate=1., bound=1.,
        )
        torch.testing.assert_close(projected, torch.tensor([[.7, .3]], dtype=torch.float64))

    def test_global_disparity_is_from_summed_confusions_not_mean_local_duals(self):
        client_zero = torch.tensor([
            [[90., 10.], [0., 0.]],
            [[10., 90.], [0., 0.]],
        ])
        client_one = torch.tensor([
            [[0., 0.], [1., 9.]],
            [[0., 0.], [9., 1.]],
        ])
        global_d = disparity_from_confusion(client_zero + client_one, "DP")
        local_ds = torch.cat([
            disparity_from_confusion(client_zero, "DP"),
            disparity_from_confusion(client_one, "DP"),
        ])
        torch.testing.assert_close(
            local_ds, torch.tensor([.8, -.8], dtype=torch.float64)
        )
        self.assertAlmostEqual(local_ds.mean().item(), 0.)
        self.assertAlmostEqual(global_d.item(), 72 / 110)
        # Dual values are deliberately unrelated; averaging them must not enter this path.
        local_duals = torch.tensor([[[4., 0.]], [[0., 4.]]], dtype=torch.float64)
        next_global = update_dual(
            torch.zeros((1, 2), dtype=torch.float64), global_d,
            tolerance=.1, learning_rate=.5, bound=5.,
        )
        self.assertFalse(torch.equal(next_global, local_duals.mean(0)))
        torch.testing.assert_close(
            next_global,
            torch.tensor([[.5 * (72 / 110 - .1), 0.]], dtype=torch.float64),
        )
