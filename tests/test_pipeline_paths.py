"""Filesystem-only pipeline checks; no Torch, model loading, jobs, or sampling."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


output_paths = module('evaluation_output_paths', ROOT / 'utils/files.py')
paths = module('repository_paths', ROOT / 'utils/paths.py')


class Broadcast:
    def __init__(self):
        self.rank = 0
        self.message = None

    def broadcast_object_list(self, message, src):
        assert src == 0
        if self.rank == 0:
            self.message = message[:]
        else:
            message[:] = self.message


class PipelinePaths(unittest.TestCase):
    def test_late_rank_accepts_files_written_by_an_earlier_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'model/seed1'
            collective = Broadcast()
            output_paths.prepare_output_directory(output, 0, collective)
            (output / 'first_bytes_rank0.npy').write_bytes(b'new run data')
            collective.rank = 1
            output_paths.prepare_output_directory(output, 1, collective)
            self.assertEqual((output / 'first_bytes_rank0.npy').read_bytes(), b'new run data')

    def test_existing_output_is_rejected_on_every_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            saved = output / 'results.json'
            saved.write_text('previous result')
            collective = Broadcast()
            for rank in (0, 1):
                collective.rank = rank
                with self.assertRaisesRegex(RuntimeError, 'Refusing to overwrite'):
                    output_paths.prepare_output_directory(output, rank, collective)
            self.assertEqual(saved.read_text(), 'previous result')

    def test_relative_and_external_paths(self):
        self.assertEqual(paths.resolve('assets/fid_stats'), ROOT/'assets/fid_stats')
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(paths.resolve(Path(folder)/'reference.npz'), Path(folder)/'reference.npz')


if __name__ == '__main__':
    unittest.main()
