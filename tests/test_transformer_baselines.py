#!/usr/bin/env python

"""Tests for `transformer_baselines` package."""


import unittest


from transformer_baselines.tasks import ClassificationTask
from transformer_baselines.tuner import Tuner


class TestTuner_basic(unittest.TestCase):
    """Tests for `transformer_baselines` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_basic_multitask_compute(self):
        """Test basic case with multitask learning."""
        labels_A = [0, 1, 0, 1, 0, 1, 0, 1]
        labels_B = [1, 0, 1, 0, 1, 0, 1, 0]

        task_A = ClassificationTask(
            model_name="bert-base-uncased",
            texts=["test", "gigi"] * 4,
            labels=labels_A,
            optimize="memory",
        )
        task_B = ClassificationTask(
            model_name="bert-base-uncased",
            texts=["test", "gigi"] * 4,
            labels=labels_B,
            optimize="memory",
        )

        t = Tuner("bert-base-uncased", "bert-base-uncased")
        t.fit(tasks=[task_A, task_B], batch_size=4)

        y_preds = t.predict(["test", "gigi"] * 4, optimize="compute")
        print(y_preds)

        self.assertEqual(
            len(y_preds), 2, "The number of label lists must be equal to tasks"
        )
        self.assertEqual(y_preds, [labels_A, labels_B])


if __name__ == "__main__":
    unittest.main()
