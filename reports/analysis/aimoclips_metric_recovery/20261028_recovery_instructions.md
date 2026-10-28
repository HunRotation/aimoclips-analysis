# AImoclips reference-manifest recovery

Recovered on 28 October 2026 (Asia/Seoul).

The manifest identifies all 2,000 retained MTG-Jamendo tracks. Regenerating the
original seeded sample exactly matches the retained directory. The package
also preserves 25 distribution-script snapshots and 12 setup-script snapshots
from editor history. An editor snapshot is not a record that the script ran.

Start with [the recovery report](20261028_metric_recovery.md).
The manifest and its validation are `20261028_reference_track_manifest.csv`
and `20261028_reference_recovery_validation.json`.

## 1. Regenerate and verify the selection

Use Python 3.10 with pandas 2.3.3 and NumPy 1.26.4, the environment used for
the verified reconstruction. Run the commands below from
`reports/analysis/aimoclips_metric_recovery/`. With the retained reference directory available:

```sh
python scripts/recover_reference_manifest.py \
  --metadata data/raw_30s_cleantags.tsv \
  --audio-dir /path/to/mtg-jamendo \
  --output recovered_manifest.csv \
  --validation recovered_validation.json
```

The script preserves the historical parsing behavior, checks the metadata
snapshot hash, verifies that the directory contains exactly the sampled tracks,
and hashes every audio file. It refuses to overwrite prior output files.
Sampling order is recovered; today's directory enumeration is recorded
separately and must not be described as a historical embedding order.

## 2. Restore reference audio when needed

```sh
python scripts/download_metric_assets.py reference \
  --manifest 20261028_reference_track_manifest.csv \
  --destination reference_audio
```

The default is a plan only. Add `--download` to fetch the original URLs and
verify every SHA-256. The full retained reference audio is approximately
21.3 GB and is not included here. The downloader preserves existing files
with different contents and stops on download/hash errors. The original URLs
were recovered from the setup code; current remote audio availability was
not tested. Changed upstream bytes will correctly fail hash verification.

## 3. Select and download the model explicitly

```sh
python scripts/download_metric_assets.py checkpoint \
  --variant music-audioset --destination checkpoints
```

Add `--download` to fetch the stated music checkpoint. The URL pins a Hugging
Face repository revision and verifies the public LFS SHA-256. Its architecture
is `HTSAT-base`. The separate `default-audioset` option is `HTSAT-tiny` and
represents the fallback seen in retained failed-run logs. It is not an
equivalent replacement for the manuscript's named music checkpoint.

Explicit music-model initialization, following the official LAION code:

```python
import laion_clap
model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
model.load_ckpt(ckpt="checkpoints/music_audioset_epoch_15_esc_90.14.pt")
model.eval()
```

The new downloader uses Python's standard library. No model or reference audio
was downloaded by the recovery work. Offline verified-download tests are
recorded in `20261028_package_validation.json`. A full network/audio/model
reproduction has not been performed.

## 4. Calculate FAD and CLAP scores as separate stages

See the report for the recovered scalar FAD implementation and the separate
per-clip CLAP-scoring procedure. The original scripts and editor snapshots
are evidence, not a newly validated executable pipeline. They contain known
fallback, path, and error-handling issues. Do not run them against existing
saved results without preparing an isolated output directory.

The seed 42 recovers track selection only. No original seed or offsets for
CLAP's random 10-second reference crops were recovered. A new fixed crop seed
would define a new calculation, not recover the original random state.

These recovery files preserve the saved metric values and statistical pipeline.
See `data/THIRD_PARTY_NOTICES.md` for third-party metadata attribution.
