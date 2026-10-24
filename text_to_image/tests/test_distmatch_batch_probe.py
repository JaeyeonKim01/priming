import unittest

from tools.probe_distmatch_batch_size import find_boundary


class BatchBoundaryTests(unittest.TestCase):
    def test_exact_adjacent_boundary_and_no_duplicate_probes(self):
        for capacity in (0, 1, 17, 31, 32, 45, 63, 64, 100, 127, 128, 300):
            tried = []
            def fits(n):
                self.assertNotIn(n, tried)
                tried.append(n)
                return n <= capacity
            self.assertEqual(find_boundary(fits), (capacity, capacity + 1))

    def test_non_memory_errors_are_not_a_capacity_limit(self):
        def broken(n):
            raise RuntimeError("not an OOM")
        with self.assertRaisesRegex(RuntimeError, "not an OOM"):
            find_boundary(broken)

    def test_missing_upper_bound_and_invalid_arguments(self):
        with self.assertRaisesRegex(RuntimeError, "no failing upper"):
            find_boundary(lambda n: True, ceiling=128)
        with self.assertRaises(ValueError):
            find_boundary(lambda n: False, start=0)
