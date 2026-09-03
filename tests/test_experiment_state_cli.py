import argparse
import unittest

from tool.experiment_cli import add_experiment_state_arguments


class ExperimentStateArgumentsTest(unittest.TestCase):
    def parse(self, values):
        parser = argparse.ArgumentParser()
        add_experiment_state_arguments(parser)
        return vars(parser.parse_args(values))

    def test_scientific_defaults(self):
        params = self.parse([])
        self.assertFalse(params["resume"])
        self.assertEqual(params["exp_repeat_times"], 3)
        self.assertEqual(params["parallel_repeats"], 1)
        self.assertEqual(params["use_amp"], "auto")
        self.assertEqual(params["base_seed"], 42)
        self.assertEqual(params["partition_min_size"], 1)
        self.assertEqual(params["partition_max_retries"], 100)
        self.assertEqual(params["partition_repair_policy"], "minimum_move_v1")
        self.assertEqual(params["partition_cache_root"], "./partition_cache")
        self.assertEqual(params["final_artifact_policy"], "metrics_only")
        self.assertEqual(params["checkpoint_keep_latest"], 1)

    def test_resume_is_explicit_and_values_are_configurable(self):
        params = self.parse([
            "-resume", "-base_seed", "9", "-exp_repeat_times", "2",
            "-partition_min_size", "3", "-partition_max_retries", "7",
            "-final_artifact_policy", "full_state", "-use_amp", "false",
        ])
        self.assertTrue(params["resume"])
        self.assertEqual(params["base_seed"], 9)
        self.assertEqual(params["exp_repeat_times"], 2)
        self.assertEqual(params["partition_min_size"], 3)
        self.assertEqual(params["partition_max_retries"], 7)
        self.assertEqual(params["final_artifact_policy"], "full_state")
        self.assertEqual(params["use_amp"], "false")


if __name__ == "__main__":
    unittest.main()
