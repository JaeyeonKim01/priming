import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from world_modeling.utils.run_control import stop_reason
from world_modeling.utils.training import record_initialization


class RunControlTests(unittest.TestCase):
    def test_deadline_margin_session_limit_and_stop_file(self):
        with patch.dict(os.environ, {'WORLD_MODEL_DEADLINE': '1000'}):
            settings = {'deadline_margin_seconds': 180}
            self.assertIsNone(stop_reason(settings, 0, now=819, monotonic=10))
            self.assertEqual(stop_reason(settings, 0, now=820, monotonic=10), 'wall_time')
            self.assertEqual(stop_reason({**settings, 'max_session_seconds': 5}, 0,
                                         now=700, monotonic=5), 'session_time')
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'stop'
                settings['stop_file'] = str(path)
                self.assertIsNone(stop_reason(settings, 0, now=700, monotonic=10))
                path.touch()
                self.assertEqual(stop_reason(settings, 0, now=700, monotonic=10), 'stop_file')

    def test_handoff_audit_and_resume_log_recovery(self):
        model = torch.nn.Module()
        model.model = torch.nn.Linear(2, 2)
        model.ema_model = copy.deepcopy(model.model)
        optimizer = torch.optim.AdamW(model.model.parameters())
        loader = torch.utils.data.DataLoader(torch.zeros(4, 2), batch_size=2)
        cfg = OmegaConf.create(dict(stage='posttrain', resume_checkpoint=None))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_initialization(root, cfg, model, optimizer, 0, {'sha256':'fixture'}, 8, loader)
            audit = json.loads((root/'initialization-step0.json').read_text())
            self.assertTrue(audit['online_equals_ema'])
            self.assertEqual(audit['optimizer_state_entries'], 0)
            with torch.no_grad(): model.model.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, 'fresh optimizer and matching'):
                record_initialization(root, cfg, model, optimizer, 0, {}, 8, loader)
            cfg.resume_checkpoint = '/checkpoint/step2.pth'
            (root/'metrics.jsonl').write_text(''.join(json.dumps({'step':s})+'\n' for s in (1,2,3)))
            record_initialization(root, cfg, model, optimizer, 2, {}, 8, loader)
            self.assertEqual([json.loads(x)['step'] for x in (root/'metrics.jsonl').read_text().splitlines()], [1,2])
            backups = list(root.glob('metrics-before-resume-*.jsonl'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(len(backups[0].read_text().splitlines()), 3)


if __name__ == '__main__': unittest.main()
