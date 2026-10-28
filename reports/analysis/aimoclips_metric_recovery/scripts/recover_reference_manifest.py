"""Recover the historical MTG-Jamendo selection and verify the retained audio set.

This intentionally preserves the historical parser, including skipped multi-tag
rows. It does not propose a corrected sample or recover historical crop offsets.
Requires pandas==2.3.3 and numpy==1.26.4 for the verified reconstruction.
"""

import argparse
from collections import Counter
import csv
import glob
import hashlib
import json
from pathlib import Path

SNAPSHOT_SHA256 = "e827347a505d6ed4e2ce213b8f1a83e5ac62e651ab7ddff426c7f04fa007d680"
BASE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/"


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd

    if args.output.exists() or args.validation.exists():
        raise FileExistsError("Choose new output paths; existing files are preserved.")
    metadata_hash = sha256(args.metadata)
    if metadata_hash != SNAPSHOT_SHA256:
        raise ValueError("The metadata differs from the recovered historical snapshot.")

    # Preserve these exact options to reconstruct the original selection.
    frame = pd.read_csv(args.metadata, sep="\t", on_bad_lines="skip", quoting=3)
    chosen = frame.sample(n=2000, random_state=42)
    observed_order = glob.glob(str(args.audio_dir / "**" / "*.mp3"), recursive=True)
    observed_order += glob.glob(str(args.audio_dir / "**" / "*.wav"), recursive=True)
    observed = {Path(path).name: Path(path) for path in observed_order}
    expected_names = [Path(str(value)).name for value in chosen["PATH"]]
    if len(observed) != len(observed_order) or len(set(expected_names)) != 2000:
        raise ValueError("Duplicate audio basenames prevent unambiguous reconstruction.")
    missing = sorted(set(expected_names) - set(observed))
    unexpected = sorted(set(observed) - set(expected_names))
    if missing or unexpected:
        raise ValueError(f"Audio set differs: missing={missing[:10]}, unexpected={unexpected[:10]}")

    raw_lines = args.metadata.read_text().splitlines()
    source_lines = {}
    for line_number, line in enumerate(raw_lines[1:], start=2):
        fields = line.split("\t")
        source_lines[Path(fields[3]).name] = line_number
    current_order = {Path(path).name: index for index, path in enumerate(observed_order, start=1)}
    records = []
    for selection_order, (row_index, row) in enumerate(chosen.iterrows(), start=1):
        name = Path(str(row["PATH"])).name
        audio = observed[name]
        records.append({
            "selection_order": selection_order,
            "source_line_number": source_lines[name],
            "parsed_row_index": int(row_index),
            "track_id": row["TRACK_ID"],
            "artist_id": row["ARTIST_ID"],
            "album_id": row["ALBUM_ID"],
            "source_relative_path": row["PATH"],
            "filename": name,
            "metadata_duration_seconds": row["DURATION"],
            "tags": row["TAGS"],
            "source_url": BASE_URL + str(row["PATH"]),
            "bytes": audio.stat().st_size,
            "sha256": sha256(audio),
            "observed_glob_order_20261028": current_order[name],
        })
        if selection_order % 250 == 0:
            print(f"Verified and hashed {selection_order}/2000 audio files", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "recovery_date_seoul": "2026-10-28",
        "metadata_sha256": metadata_hash,
        "metadata_raw_data_rows": len(raw_lines) - 1,
        "raw_field_count_histogram": dict(Counter(len(line.split("\t")) for line in raw_lines[1:])),
        "historical_parser": "pd.read_csv(path, sep='\\t', on_bad_lines='skip', quoting=3)",
        "parsed_eligible_rows": len(frame),
        "skipped_rows": len(raw_lines) - 1 - len(frame),
        "selection": "frame.sample(n=2000, random_state=42)",
        "sampling_without_replacement": True,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "selected_track_count": len(records),
        "retained_track_count": len(observed),
        "exact_track_set_match": True,
        "total_audio_bytes": sum(row["bytes"] for row in records),
        "manifest_sha256": sha256(args.output),
        "missing_tracks": missing,
        "unexpected_tracks": unexpected,
        "historical_fad_reproduced": False,
        "historical_checkpoint_proven_for_saved_scores": False,
        "historical_crop_offsets_recovered": False,
        "order_note": "selection_order is the recovered pandas selection order; observed_glob_order is today's filesystem enumeration, not a recovered historical embedding order.",
        "checksum_note": "Audio hashes describe the retained files on the recovery date; no historical audio checksums were found for independent comparison.",
    }
    args.validation.parent.mkdir(parents=True, exist_ok=True)
    with args.validation.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
