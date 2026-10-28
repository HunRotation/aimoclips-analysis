"""Plan or download reference audio and an explicitly selected CLAP checkpoint.

Downloads require --download. Every existing or downloaded file is checked
against its SHA-256. This script never substitutes a different checkpoint.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "b3708341862f581175dba5c356a4ebf74a9b6651"
CHECKPOINTS = {
    "music-audioset": {
        "filename": "music_audioset_epoch_15_esc_90.14.pt",
        "sha256": "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd",
        "bytes": 2352471003,
        "amodel": "HTSAT-base",
    },
    "default-audioset": {
        "filename": "630k-audioset-best.pt",
        "sha256": "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037",
        "bytes": 1863587645,
        "amodel": "HTSAT-tiny",
    },
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch_verified(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Existing file has different contents: {destination}")
        return "already verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    # Preserve incomplete files from previous invocations for inspection.
    with temporary.open("xb") as output:
        with urllib.request.urlopen(url, timeout=120) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
    if sha256(temporary) != expected_sha256:
        raise ValueError(f"Downloaded bytes differ from the manifest: {temporary}")
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    return "downloaded and verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="asset", required=True)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--variant", choices=CHECKPOINTS, required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--manifest", type=Path, required=True)
    for subparser in [checkpoint, reference]:
        subparser.add_argument("--destination", type=Path, required=True)
        subparser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.asset == "checkpoint":
        record = CHECKPOINTS[args.variant].copy()
        record["url"] = (
            "https://huggingface.co/lukewys/laion_clap/resolve/"
            + REVISION + "/" + record["filename"]
        )
        record["huggingface_revision"] = REVISION
        print(json.dumps(record, indent=2))
        if args.download:
            print(fetch_verified(record["url"], args.destination / record["filename"], record["sha256"]))
    else:
        with args.manifest.open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if len(records) != 2000 or len({row["filename"] for row in records}) != 2000:
            raise ValueError("The recovered manifest must contain 2000 unique tracks.")
        for row in records:
            if Path(row["filename"]).name != row["filename"]:
                raise ValueError("Invalid reference basename")
            if not row["source_url"].startswith("https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/"):
                raise ValueError("Unexpected reference source URL")
        print(json.dumps({"tracks": len(records), "bytes": sum(int(row["bytes"]) for row in records), "destination": str(args.destination), "download": args.download}, indent=2))
        if args.download:
            for index, row in enumerate(records, start=1):
                status = fetch_verified(row["source_url"], args.destination / row["filename"], row["sha256"])
                print(f"{index}/{len(records)} {row['filename']}: {status}", flush=True)


if __name__ == "__main__":
    main()
