"""Integrity and portability failures must stop asset preparation."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import download_assets as downloads
import publish_dataset as publisher


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = b'fixed reference data'
        self.row = dict(path='assets/reference.bin', bytes=len(self.data),
                        sha256=hashlib.sha256(self.data).hexdigest(), kind='url',
                        url='https://example.invalid/reference.bin')

    def test_verify_existing_never_uses_network(self):
        target = self.root / self.row['path']
        target.parent.mkdir(); target.write_bytes(self.data)
        with patch.object(downloads.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            downloads.download(self.root, self.row, verify_only=True)

    def test_corrupt_existing_file_is_not_overwritten(self):
        target = self.root / self.row['path']
        target.parent.mkdir(); target.write_bytes(b'x' * len(self.data))
        with self.assertRaises(ValueError), patch.object(downloads.urllib.request, 'urlopen') as request:
            downloads.download(self.root, self.row)
        request.assert_not_called()
        self.assertEqual(target.read_bytes(), b'x' * len(self.data))

    def test_corrupt_download_does_not_become_final_file(self):
        with patch.object(downloads.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong')):
            with self.assertRaises(ValueError):
                downloads.download(self.root, self.row)
        self.assertFalse((self.root / self.row['path']).exists())

    def test_missing_file_in_verify_mode_does_not_download(self):
        with self.assertRaises(FileNotFoundError), patch.object(downloads.urllib.request, 'urlopen') as request:
            downloads.download(self.root, self.row, verify_only=True)
        request.assert_not_called()

    def test_paths_cannot_escape_destination(self):
        for name in ('../outside', '/outside', 'assets/../../outside'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                downloads.safe_path(self.root, name)

    def test_conflicting_offline_revision_is_rejected(self):
        row = dict(self.row, kind='hub_cache', repo_id='test/encoder', revision='fixed')
        target = self.root / row['path']; target.parent.mkdir(); target.write_bytes(self.data)
        ref = self.root / 'assets/huggingface/hub/models--test--encoder/refs/main'
        ref.parent.mkdir(parents=True); ref.write_text('different')
        with self.assertRaises(ValueError):
            downloads.download(self.root, row)
        self.assertEqual(ref.read_text(), 'different')


class ArrowTests(unittest.TestCase):
    def shard(self, folder, image_path=None, label=0):
        import pyarrow as pa
        from PIL import Image
        image = io.BytesIO(); Image.new('RGB', (256, 256)).save(image, format='PNG')
        table = pa.table({'id': pa.array([0], type=pa.int32()),
                          'image': pa.array([dict(bytes=image.getvalue(), path=image_path)],
                                            type=pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
                          'label': pa.array([label], type=pa.int32())})
        path = Path(folder) / 'data.arrow'
        with path.open('wb') as handle, pa.ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table)
        return path

    def test_embedded_images_are_portable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(publisher.validate_arrow(self.shard(folder)).tolist(), [0])

    def test_source_paths_are_not_published(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            publisher.validate_arrow(self.shard(folder, image_path='source/image.png'))

    def test_out_of_range_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            publisher.validate_arrow(self.shard(folder, label=1000))


if __name__ == '__main__':
    unittest.main()
