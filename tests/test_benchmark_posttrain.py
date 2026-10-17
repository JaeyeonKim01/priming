"""The benchmark cannot accidentally launch an unbounded production run."""
import sys
import unittest
from unittest.mock import patch

import posttrain
from scripts import benchmark_posttrain


class BenchmarkTests(unittest.TestCase):
    def test_forwards_a_bounded_run_and_restores_instrumentation(self):
        original = (posttrain.train_step, posttrain.optimizer_step, posttrain.run_training)
        arguments = ['benchmark', '--steps', '130', '--', '--config', 'fast.yaml', '--output', 'run']
        with patch.object(sys, 'argv', arguments), patch.object(posttrain, 'main', return_value=0) as main:
            self.assertEqual(benchmark_posttrain.main(), 0)
        main.assert_called_once_with(['--config', 'fast.yaml', '--output', 'run', '--smoke-steps', '130'])
        self.assertEqual(original, (posttrain.train_step, posttrain.optimizer_step, posttrain.run_training))

    def test_rejects_unbounded_or_overridden_step_counts(self):
        for arguments in (['--steps', '0'], ['--steps', '201'], ['--', '--smoke-steps', '25000'],
                          ['--', '--smoke-steps=25000']):
            with self.subTest(arguments=arguments), patch.object(sys, 'argv', ['benchmark', *arguments]), \
                    patch.object(posttrain, 'main') as main, patch('sys.stderr'), self.assertRaises(SystemExit):
                benchmark_posttrain.main()
            main.assert_not_called()


if __name__ == '__main__':
    unittest.main()
