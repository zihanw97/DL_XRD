#!/usr/bin/env python3
"""Download the checksummed input-only GitHub release; never starts training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "zihanw97/DL_XRD"
RELEASE_TAG = "manuscript-clean-inputs-20261106"
RAW_ROOTS = {"sim_dataset_new", "experiment_1500_maxima", "experiment_1500_maxima_outliner"}
CACHE_PATHS = {
    "fixed_images.npz": "data/cache/fixed_images.npz",
    "xrd_dataset_cache.npz": "data/cache/xrd_dataset_cache.npz",
}
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size == size and sha256_file(path) == digest


def validate_manifest(document: dict) -> list[dict]:
    """Accept only the declared repository/tag, cache destinations and part sequence."""
    if document.get("schema") != 1 or document.get("repository") != REPOSITORY or document.get("tag") != RELEASE_TAG:
        raise ValueError("Unexpected input-release schema, repository, or tag")
    assets = document.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("Input release has no assets")
    names: set[str] = set()
    caches: set[str] = set()
    parts: list[int] = []
    for asset in assets:
        name = asset.get("name", "")
        if not isinstance(name, str) or name in names or "/" in name or "\\" in name:
            raise ValueError("Unsafe or duplicate asset name")
        names.add(name)
        if not isinstance(asset.get("size"), int) or isinstance(asset["size"], bool) or asset["size"] <= 0:
            raise ValueError(f"Invalid asset size: {name}")
        if not isinstance(asset.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
            raise ValueError(f"Invalid SHA256: {name}")
        if asset.get("url") != f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{name}":
            raise ValueError(f"Unexpected asset URL: {name}")
        if asset.get("kind") == "cache":
            if name not in CACHE_PATHS or asset.get("destination") != CACHE_PATHS[name]:
                raise ValueError(f"Unexpected cache destination: {name}")
            caches.add(name)
        elif asset.get("kind") == "raw_archive_part":
            match = re.fullmatch(r"raw_inputs\.tar\.gz\.part([0-9]{2})", name)
            if not match or "destination" in asset:
                raise ValueError(f"Unexpected raw archive part: {name}")
            parts.append(int(match.group(1)))
        else:
            raise ValueError(f"Unexpected asset kind: {name}")
    if caches != set(CACHE_PATHS) or not parts or sorted(parts) != list(range(len(parts))):
        raise ValueError("Both caches and a consecutive raw archive part sequence are required")
    return assets


def checked_target(root: Path, relative: str) -> Path:
    """Refuse absolute paths, traversal, and pre-existing symlink components."""
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError(f"Unsafe input path: {relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        raise ValueError(f"Unsafe input path: {relative!r}")
    target = root.joinpath(*parsed.parts)
    cursor = root
    for component in parsed.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"Refusing symlink in input path: {relative}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Input path leaves package: {relative}")
    return target


def fetch_asset(asset: dict, download_dir: Path, *, force: bool = False) -> Path:
    """Verify local copies or stream into a temporary file before atomic replacement."""
    destination = checked_target(download_dir, asset["name"])
    if matches(destination, asset["size"], asset["sha256"]):
        print(f"Verified download: {asset['name']}", flush=True)
        return destination
    if destination.exists() and not force:
        raise ValueError(f"Existing download is corrupt: {destination}; use --force to replace it")
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB)", flush=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{asset['name']}.", suffix=".partial", dir=download_dir)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        request = Request(asset["url"], headers={"User-Agent": "DL-XRD-input-downloader/1", "Accept": "application/octet-stream"})
        with os.fdopen(descriptor, "wb") as output, urlopen(request, timeout=120) as response:
            while chunk := response.read(CHUNK):
                size += len(chunk)
                if size > asset["size"]:
                    raise ValueError(f"Download exceeds declared size: {asset['name']}")
                digest.update(chunk)
                output.write(chunk)
        if size != asset["size"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError(f"Downloaded size or SHA256 mismatch: {asset['name']}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def copy_cache(source: Path, destination: Path, asset: dict, *, force: bool = False) -> None:
    if matches(destination, asset["size"], asset["sha256"]):
        return
    if destination.exists() and not force:
        raise ValueError(f"Existing input is corrupt: {destination}; use --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output, CHUNK)
        if not matches(temporary, asset["size"], asset["sha256"]):
            raise ValueError(f"Cache copy failed verification: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def raw_inventory(root: Path) -> dict[str, tuple[int, str]]:
    with (root / "data/metadata/raw_file_hashes.csv").open(newline="", encoding="utf-8") as stream:
        return {row["path"]: (int(row["bytes"]), row["sha256"]) for row in csv.DictReader(stream)}


def check_raw(root: Path, inventory: dict[str, tuple[int, str]]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    invalid: list[str] = []
    for relative, (size, digest) in inventory.items():
        destination = checked_target(root, relative)
        if not destination.exists():
            missing.append(relative)
        elif not matches(destination, size, digest):
            invalid.append(relative)
    return missing, invalid


def reconstruct_archive(parts: list[Path], destination: Path) -> Path:
    """Combine already verified pieces without loading the archive into memory."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".raw_inputs.", suffix=".partial", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for part in parts:
                with part.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output, CHUNK)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_raw_archive(archive: Path, root: Path, inventory: dict[str, tuple[int, str]], *, force: bool = False) -> None:
    """Validate every header before writing; accept only ordinary raw input files."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names: set[str] = set()
        for member in members:
            destination = checked_target(root, member.name)
            path = PurePosixPath(member.name)
            if len(path.parts) < 2 or path.parts[0] != "data" or path.parts[1] not in RAW_ROOTS:
                raise ValueError(f"Archive member is outside the raw input directories: {member.name}")
            if member.name in names:
                raise ValueError(f"Duplicate archive member: {member.name}")
            names.add(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Archive links/devices are forbidden: {member.name}")
            if member.isfile():
                if member.name not in inventory and path.name != "instructions.txt":
                    raise ValueError(f"Unregistered raw input file: {member.name}")
                if member.name in inventory and member.size != inventory[member.name][0]:
                    raise ValueError(f"Archive size differs from raw inventory: {member.name}")
                if destination.exists() and not destination.is_file():
                    raise ValueError(f"Raw input target is not a regular file: {member.name}")
        missing_members = set(inventory) - names
        if missing_members:
            raise ValueError(f"Archive omits {len(missing_members)} registered raw inputs")
        for member in members:
            destination = checked_target(root, member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            expected = inventory.get(member.name)
            if expected and matches(destination, *expected):
                continue
            if destination.exists() and not force:
                # Existing instructions are preserved when their content agrees.
                if expected:
                    raise ValueError(f"Existing raw input is corrupt: {member.name}; use --force to replace it")
                with bundle.extractfile(member) as incoming:
                    if destination.read_bytes() == incoming.read():
                        continue
                raise ValueError(f"Existing acquisition instructions differ: {member.name}; use --force to replace them")
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output, bundle.extractfile(member) as incoming:
                    shutil.copyfileobj(incoming, output, CHUNK)
                if expected and not matches(temporary, *expected):
                    raise ValueError(f"Extracted raw SHA256 mismatch: {member.name}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Download only the two image/label caches, without raw TIFFs")
    parser.add_argument("--verify-only", action="store_true", help="Check installed inputs without any network requests")
    parser.add_argument("--download-dir", type=Path, default=PACKAGE_ROOT / ".input_downloads", help="Keep verified downloads here for reuse")
    parser.add_argument("--force", action="store_true", help="Replace existing inputs/downloads only after the replacement passes checksum verification")
    args = parser.parse_args(argv)
    manifest_path = PACKAGE_ROOT / "data/release_inputs.json"
    if not manifest_path.is_file():
        parser.error("The matching input release manifest is not installed. "
                     "This GitHub revision contains code only; see data/README.md "
                     "for input-data publication status and local setup.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = validate_manifest(manifest)
    caches = [asset for asset in assets if asset["kind"] == "cache"]
    for asset in caches:
        destination = checked_target(PACKAGE_ROOT, asset["destination"])
        if matches(destination, asset["size"], asset["sha256"]):
            print(f"Verified input: {asset['destination']}", flush=True)
            continue
        if args.verify_only:
            raise ValueError(f"Missing or corrupt input: {asset['destination']}")
        if destination.exists() and not args.force:
            raise ValueError(f"Existing input is corrupt: {destination}; use --force to replace it")
        downloaded = fetch_asset(asset, args.download_dir.resolve(), force=args.force)
        copy_cache(downloaded, destination, asset, force=args.force)
    if not args.cache_only:
        inventory = raw_inventory(PACKAGE_ROOT)
        missing, invalid = check_raw(PACKAGE_ROOT, inventory)
        if missing or invalid:
            if args.verify_only:
                raise ValueError(f"Raw inputs: {len(missing)} missing, {len(invalid)} corrupt")
            if invalid and not args.force:
                raise ValueError(f"{len(invalid)} existing raw inputs are corrupt; use --force to replace them")
            parts = sorted((asset for asset in assets if asset["kind"] == "raw_archive_part"), key=lambda asset: asset["name"])
            downloaded = [fetch_asset(asset, args.download_dir.resolve(), force=args.force) for asset in parts]
            archive = reconstruct_archive(downloaded, args.download_dir.resolve() / "raw_inputs.tar.gz")
            print("Extracting verified raw input archive", flush=True)
            extract_raw_archive(archive, PACKAGE_ROOT, inventory, force=args.force)
            missing, invalid = check_raw(PACKAGE_ROOT, inventory)
            if missing or invalid:
                raise ValueError(f"Raw-input verification failed: {len(missing)} missing, {len(invalid)} corrupt")
        print(f"Verified {len(inventory)} raw TIFF inputs", flush=True)
    print("Input setup complete. No training was started.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tarfile.TarError) as error:
        raise SystemExit(f"Input setup failed: {error}") from error
