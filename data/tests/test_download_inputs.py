"""Input release integrity and safe extraction checks; all fixtures are temporary."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from data.download_inputs import (
    REPOSITORY, RELEASE_TAG, CACHE_PATHS, checked_target, copy_cache,
    extract_raw_archive, fetch_asset, matches, reconstruct_archive, validate_manifest,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def asset(name, payload, kind="cache", destination=None):
    result = dict(name=name, size=len(payload), sha256=digest(payload), kind=kind,
                  url=f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{name}")
    if destination is not None:
        result["destination"] = destination
    return result


class InputDownloadTests(unittest.TestCase):
    def test_manifest_rejects_remote_or_destination_substitution(self):
        assets = [asset(name, b"cache", destination=path) for name, path in CACHE_PATHS.items()]
        assets.append(asset("raw_inputs.tar.gz.part00", b"part", "raw_archive_part"))
        manifest = dict(schema=1, repository=REPOSITORY, tag=RELEASE_TAG, assets=assets)
        validate_manifest(manifest)
        assets[0]["url"] = "https://example.com/cache"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)
        assets[0]["url"] = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{assets[0]['name']}"
        assets[0]["destination"] = "models.py"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_corrupt_download_never_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "fixed_images.npz"
            target.write_bytes(b"old")
            metadata = asset(target.name, b"correct", destination=CACHE_PATHS[target.name])
            with patch("data.download_inputs.urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    fetch_asset(metadata, root, force=True)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_verified_download_and_cache_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = asset("fixed_images.npz", b"valid cache", destination=CACHE_PATHS["fixed_images.npz"])
            with patch("data.download_inputs.urlopen", return_value=io.BytesIO(b"valid cache")) as network:
                source = fetch_asset(metadata, root / "downloads")
                destination = checked_target(root, metadata["destination"])
                copy_cache(source, destination, metadata)
                self.assertTrue(matches(destination, metadata["size"], metadata["sha256"]))
                fetch_asset(metadata, root / "downloads")
                self.assertEqual(network.call_count, 1)

    def make_archive(self, path, members):
        with tarfile.open(path, "w:gz") as bundle:
            for name, payload, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.REGTYPE:
                    info.size = len(payload)
                    bundle.addfile(info, io.BytesIO(payload))
                else:
                    info.linkname = "../../outside"
                    bundle.addfile(info)

    def test_reassembly_and_normal_raw_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = "data/sim_dataset_new/BCC/test.tiff"
            instruction = "data/experiment_1500_maxima/sample/instructions.txt"
            archive = root / "fixture.tar.gz"
            self.make_archive(archive, [(raw, b"pixels", tarfile.REGTYPE), (instruction, b"settings", tarfile.REGTYPE)])
            compressed = archive.read_bytes()
            first, second = root / "part00", root / "part01"
            first.write_bytes(compressed[:10]); second.write_bytes(compressed[10:])
            combined = reconstruct_archive([first, second], root / "combined.tar.gz")
            self.assertEqual(combined.read_bytes(), compressed)
            output = root / "package"
            output.mkdir()
            extract_raw_archive(combined, output, {raw: (6, digest(b"pixels"))})
            self.assertEqual((output / raw).read_bytes(), b"pixels")
            self.assertEqual((output / instruction).read_bytes(), b"settings")
            extract_raw_archive(combined, output, {raw: (6, digest(b"pixels"))})

    def test_malicious_archive_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = "data/sim_dataset_new/test.tiff"
            for name, kind in [("../outside", tarfile.REGTYPE),
                               ("data/sim_dataset_new/link", tarfile.SYMTYPE),
                               ("data/sim_dataset_new/hardlink", tarfile.LNKTYPE),
                               ("data/sim_dataset_new/device", tarfile.CHRTYPE),
                               ("models.py", tarfile.REGTYPE)]:
                archive = root / "bad.tar.gz"
                self.make_archive(archive, [(raw, b"pixels", tarfile.REGTYPE), (name, b"bad", kind)])
                output = root / "package"
                output.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    extract_raw_archive(archive, output, {raw: (6, digest(b"pixels"))})
                self.assertFalse((output / raw).exists())

    def test_corrupt_raw_member_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = "data/sim_dataset_new/test.tiff"
            archive = root / "bad-pixels.tar.gz"
            self.make_archive(archive, [(raw, b"broken", tarfile.REGTYPE)])
            output = root / "package"
            target = output / raw
            target.parent.mkdir(parents=True)
            target.write_bytes(b"old")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                extract_raw_archive(archive, output, {raw: (6, digest(b"pixels"))}, force=True)
            self.assertEqual(target.read_bytes(), b"old")

    def test_existing_symlink_component_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                checked_target(root, "data/sim_dataset_new/test.tiff")


if __name__ == "__main__":
    unittest.main()
