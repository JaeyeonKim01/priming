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


output_paths = module('evaluation_output_paths', ROOT / 'eval/output_paths.py')
paths = module('training_paths', ROOT / 'training/scripts/paths.py')


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

    def test_asset_verification_follows_configured_roots(self):
        config = paths.load_config('b160')
        site = paths.site_config()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            site.update(hf_home=str(base / 'hf'), torch_home=str(base / 'torch'),
                        fid_stats=str(base / 'stats'), data_root=str(base / 'data'))
            config['data']['data_path'] = str(base / 'imagenet')
            config['ot'].update(ot_repr_cache=str(base / 'representations'),
                                ot_feature_asset_cache=str(base / 'features'),
                                ot_feature_calibration_dir=str(base / 'calibration'))
            cases = {
                'assets/huggingface/hub/model/config.json': 'hf/hub/model/config.json',
                'assets/torch/hub/checkpoints/inception.pth': 'torch/hub/checkpoints/inception.pth',
                'assets/fid_stats/reference.npz': 'stats/reference.npz',
                'data/imagenet/imagenet-latents-images/state.json': 'imagenet/imagenet-latents-images/state.json',
                'data/prc/reference.pt': 'data/prc/reference.pt',
                'assets/ot/representations/mae_encoder.pt': 'representations/mae_encoder.pt',
                'assets/ot/features/calibrations/stack.json': 'calibration/stack.json',
                'assets/ot/features/model.pt': 'features/model.pt',
            }
            for canonical, relocated in cases.items():
                self.assertEqual(paths.resolve_asset(canonical, config, site), base / relocated)
            self.assertEqual(paths.resolve_asset(base / 'explicit.pt', config, site), base / 'explicit.pt')

    def test_default_asset_locations_remain_repository_relative(self):
        for arm in ('b160', 'l160'):
            config, site = paths.load_config(arm), paths.site_config()
            for path in ('assets/fid_stats/reference.npz', 'data/imagenet/imagenet-latents-images',
                         'assets/ot/representations/calibration.json', 'data/prc/reference.pt'):
                self.assertEqual(paths.resolve_asset(path, config, site), ROOT / path)


if __name__ == '__main__':
    unittest.main()
