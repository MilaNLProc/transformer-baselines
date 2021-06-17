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
        labels_A = [0, 1, 0, 1, 0, 1, 0, 1]
        labels_B = [1, 0, 1, 0, 1, 0, 1, 0]

        task_A = ClassificationTask(
            texts=["Hello I like this very much", "I am too sad"] * 4,
            labels=labels_A,
        )
        task_B = ClassificationTask(
            texts=["Hello I like this", "I am very sad"] * 4,
            labels=labels_B,
        )

        t = Tuner("bert-base-uncased", "bert-base-uncased")
        t.fit(tasks=[task_A, task_B], batch_size=4, max_epochs=20)

        y_pred = t.predict(["Hello I like this very much", "I am too sad"] * 4, optimize="compute")
        self.assertEqual(
             len(y_pred), 2, "The number of label lists must be equal to tasks"
         )

        y_pred_1 = t.predict(["Hello I like this very much", "I am too sad"] * 4, optimize="compute")[0]

        y_pred_2 = t.predict(["Hello I like this", "I am very sad"] * 4, optimize="compute")[1]

        self.assertEqual([y_pred_1, y_pred_2], [labels_A, labels_B])


if __name__ == "__main__":
    unittest.main()
