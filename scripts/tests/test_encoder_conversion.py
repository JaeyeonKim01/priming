"""Encoder packaging must preserve payloads and enforce the release file hash."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_ot_encoders import normalize_serialization_id, save_verified_encoder


class EncoderPackagingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.state = {'weight': torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        self.reference = self.root / 'encoder.pt'
        torch.save(self.state, self.reference)
        with zipfile.ZipFile(self.reference) as archive:
            self.identity = archive.read('encoder/.data/serialization_id').decode()
            self.payloads = {name: archive.read(name) for name in archive.namelist()
                             if not name.endswith('/.data/serialization_id')}
        self.expected = dict(bytes=self.reference.stat().st_size,
                             sha256=hashlib.sha256(self.reference.read_bytes()).hexdigest(),
                             serialization_id=self.identity)

    def test_normalization_changes_only_metadata_and_preserves_loading(self):
        original = self.reference.read_bytes()
        normalize_serialization_id(self.reference, '0' * 40)
        with zipfile.ZipFile(self.reference) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read('encoder/.data/serialization_id'), b'0' * 40)
            for name, data in self.payloads.items():
                self.assertEqual(archive.read(name), data)
        loaded = torch.load(self.reference, weights_only=True)
        self.assertTrue(torch.equal(loaded['weight'], self.state['weight']))
        normalize_serialization_id(self.reference, self.identity)
        self.assertEqual(self.reference.read_bytes(), original)

    def test_matching_encoder_is_published_with_exact_release_bytes(self):
        output = self.root / 'converted' / 'encoder.pt'
        save_verified_encoder(self.state, output, self.expected)
        self.assertEqual(output.read_bytes(), self.reference.read_bytes())
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_changed_tensor_is_rejected_without_leaving_an_output(self):
        output = self.root / 'converted' / 'encoder.pt'
        with self.assertRaisesRegex(ValueError, 'differs from the pinned release'):
            save_verified_encoder({'weight': self.state['weight'] + 1}, output, self.expected)
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_existing_output_is_not_overwritten(self):
        before = self.reference.read_bytes()
        with self.assertRaises(FileExistsError):
            save_verified_encoder(self.state, self.reference, self.expected)
        self.assertEqual(self.reference.read_bytes(), before)

    def test_invalid_identity_is_rejected(self):
        for identity in ('123', 'x' * 40, '\u0661' * 40):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                normalize_serialization_id(self.reference, identity)


if __name__ == '__main__':
    unittest.main()
