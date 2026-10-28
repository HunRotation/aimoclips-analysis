# AImoclips: reference selection, FAD, and CLAP recovery

28 October 2026, Asia/Seoul.

The 2,000-track reference selection has been recovered and independently
checked against the retained audio directory. The exact historical FAD
calculation has not been reproduced. These are distinct results: recovering
track identities does not recover model choices, random crop positions, or
an authenticated successful run producing the six saved FAD values.

## Recovered reference selection

The original `analysis/setup_metrics_env.py` and its editor-history copies
contain the following selection:

```python
df = pd.read_csv(tsv_path, sep="\t", on_bad_lines="skip", quoting=3)
sample = df.sample(n=2000, random_state=42)
```

The retained snapshot is `data/raw_30s_cleantags.tsv`, SHA-256
`e827347a505d6ed4e2ce213b8f1a83e5ac62e651ab7ddff426c7f04fa007d680`.
It has 55,701 data rows. Because multiple tags occupy additional tab-separated
fields, the historical parser accepts the 5,107 rows with exactly six fields
and skips the other 50,594. The actual reference pool is therefore a
single-tag subset, not an unrestricted sample of all MTG-Jamendo tracks.
The recovered procedure intentionally preserves this behavior. Correcting
the parser would change the sampling pool and require a new metric analysis.

With pandas 2.3.3 and NumPy 1.26.4, sampling 2,000 rows without replacement
with `random_state=42` matches all 2,000 MP3 filenames in the retained directory:
2,000 matches, zero missing, and zero unexpected files. The manifest includes
track, artist and album identifiers, the metadata line, original relative path,
source URL, original selection order, current file size, and SHA-256. The
audio bytes total 21,275,847,215. The checksums were generated now; no prior
audio hashes were found to prove that the files are unchanged since the run.

The original setup code downloads the sampled files concurrently and flattens
their basenames into a directory. The distribution script later enumerates
MP3s and WAVs using an unsorted `glob`. It only makes an additional unseeded
NumPy selection when more than 2,000 files are present. That branch is not
needed for the recovered directory, which contains exactly 2,000 MP3s. This
explains why a fixed setup seed can recover the selection even though the
visible FAD script contains an unseeded `np.random.choice` branch.

`selection_order` is the recovered pandas order. The manifest's current-glob
column is a present-day observation. No historical extraction order or random
audio-crop offsets have been established. The `raw_30s` name means tracks
longer than 30 seconds, not files cropped to 30 seconds. Header inspection of
the first three selected files found full tracks approximately 345.47,
280.35 and 269.71 seconds long, respectively.

## Recovered FAD calculation process

Editor history contains 25 versions of `calculate_distribution_metrics.py`.
Nineteen are scalar variants and six are later bootstrap variants. The live
script is one of the bootstrap variants, which explains why inspecting it
alone did not identify the writer of `distribution_metrics.csv`.

The scalar variants recover this procedure:

1. Load one CLAP audio embedding per reference track and calculate its mean
   vector and sample covariance (`np.cov(..., rowvar=False)`).
2. Enumerate audio in each generated-system directory and obtain its CLAP
   audio embeddings. Calculate that system's mean and sample covariance.
3. Calculate the squared mean-vector difference plus the two covariance
   traces, minus twice the trace of the matrix square root of their product:
   `||mu_system - mu_reference||^2 + tr(S_system + S_reference
   - 2 * sqrtm(S_system @ S_reference))`.
4. Write one `fad_score` per system into `distribution_metrics.csv`.

The source uses SciPy `sqrtm`, a small diagonal regularization fallback for
nonfinite results, and a check for numerical imaginary components. Recovered
examples are `source/editor_history/distribution/88dC.py` (an explicit scalar
writer) and `jupP.py` (a later version that can retain existing CSV values and
only calculate missing ones). This caching behavior means a later successful
save can preserve FAD values from an earlier unrecorded calculation.

The scalar script enumerates system folders rather than explicitly defining
the retained 991-clip analysis set. A future metric rerun must record its
generated-track membership, rather than assume it matches the current ratings.
It must also fail on embedding errors instead of silently dropping failed
batches, as the historical helper can do.

No new FAD or CLAP scores have been substituted for the saved values. The
saved six-system CSV and per-clip CLAP CSV are included as comparison inputs.

## Checkpoint and random-crop uncertainty

The intended manuscript music checkpoint is
`music_audioset_epoch_15_esc_90.14.pt`, which the official LAION instructions
pair with `HTSAT-base`, non-fusion. Several recovered scripts use that
architecture; other scalar and bootstrap versions instead instantiate
`HTSAT-tiny`. Both families fall back to `model.load_ckpt()` when a relative
checkpoint path is absent. That call chooses `630k-audioset-best.pt` for a
non-fusion model, not the music checkpoint.

The original setup downloader writes the music checkpoint below `AImoclips/`,
while the scoring scripts check for its basename in the working directory.
Those paths need not refer to the same file. This can explain fallback, but
the historical working directory is not independently proven for every run.

The retained `630k-audioset-best.pt` was hashed and exactly matches the public
default-checkpoint digest recorded in `source/checkpoint_inventory.json`.
The named music checkpoint was not recovered in the inspected local model
locations. The four retained logs show fallback and failed/interrupted CLAP
extraction attempts. Two logs contain 125 batch errors each; another has an
architecture mismatch, and another ends with a keyboard interruption. They
are not successful-run proof of the saved numeric FAD values. Author
confirmation of the intended music model remains separate evidence and is
not replaced by conclusions about these failed attempts.

The installed LAION-CLAP 1.1.7 file-list API loads mono audio at 48 kHz,
performs an int16 quantization round trip, and uses non-fusion `rand_trunc`
preprocessing with 480,000 samples. For audio longer than ten seconds,
`training/data.py` draws a crop start with `np.random.randint`. No historical
global NumPy seed, saved random-generator state, or crop-start manifest was
found in the relevant source and editor snapshots. The setup's
`random_state=42` is a local pandas sampling argument; it does not fix these
subsequent crops. A new crop seed cannot reconstruct an unrecorded past state.

A defensible new metric run should name the model and checkpoint hash,
load the explicit checkpoint without fallback, pin preprocessing versions,
read track membership from manifests, and log the precise 48-kHz crop start
and length for every reference track. Save generated-audio hashes, embeddings,
reference/system moments, and results together. Compare new scores against
the retained six values before claiming exact reproduction. A discrepancy
would require investigation and possibly updated analyses, not relabeling.

## CLAP scores are a different calculation

The recovered `calculate_clap_score.py` obtains audio embeddings for each
generated clip and a text embedding for the bare target emotion word. It
L2-normalizes each embedding and writes their dot product as the clip's raw
cosine similarity. It does not calculate FAD and does not use the Jamendo
reference tracks. It requests `HTSAT-base`, but has the same relative-path
checkpoint fallback described above. Its input mapping also strips the
`_crop` filename suffix. The exact audio bytes and input list must accompany
any new score computation.

Both FAD and CLAP similarity can use the same explicitly selected CLAP model;
only FAD compares two audio-embedding distributions. The model download does
not need to occur twice if both stages use the identical checkpoint.

## What download code is available

Before this recovery publication, the analysis repository at commit
`9bc0730926742981ac4e0f319b12d8b369cb70cf` reproduced statistics from saved
CLAP/FAD inputs. Its statistical entry point and requirements remain unchanged.
This recovery directory adds the recovered manifests, source records and separate
asset-download utilities, rather than a validated upstream metric pipeline.

The original workspace does contain `setup_metrics_env.py` with a music-model
download URL. Its current `main()` downloads that checkpoint and prepares
FMA-Pop, while the MTG-Jamendo setup call is commented out. It is not a complete
ready-to-run FAD reference setup in its present form.

This recovery package adds a separate, conservative download utility. It
requires an explicit checkpoint choice, pins the Hugging Face repository
revision `b3708341862f581175dba5c356a4ebf74a9b6651`, and checks SHA-256 values.
For reference audio, it downloads the recovered URLs rather than resampling
from current metadata. Downloads are opt-in. No model weights or reference
audio are bundled or were newly downloaded. Full remote download and model
inference have not been tested in this task.

The checkpoint initializer may additionally fetch the `roberta-base` tokenizer
and text-encoder configuration/weights through Transformers when they are
not cached. This recovery does not pin a historical Hugging Face snapshot
for those auxiliary artifacts; an offline complete pipeline would need to
cache and record them as well.

## Validation and interpretation

The manifest recovery validates all 2,000 track identities and retained-file
hashes. The separate package-validation record covers source-copy integrity,
manifest/source-row matching, a smoke test of the recovered Fréchet function,
and local verified-download behavior. These checks do not run the CLAP models
or establish equality with the saved FAD scores.

The earlier statement that no historical reference selection was recoverable
was too strong. The selection and its fixed seed are now recovered. What
remains unresolved is the end-to-end origin of the retained metric values.
The repository README and input provenance now distinguish recovered track
selection from unresolved historical metric reproduction. The publication adds
these recovery records and utilities. No manuscript, statistical analysis
script or saved metric value is changed.

## Sources and provenance

- Original workspace scripts, retained reference audio, metadata snapshot,
  local run logs, and editor history, preserved or summarized in this package.
- [LAION-CLAP official checkpoint guidance](https://github.com/LAION-AI/CLAP#pretrained-models).
- [LAION-CLAP file-list and default-model implementation](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/hook.py).
- [MTG-Jamendo structure, downloads and licenses](https://github.com/MTG/mtg-jamendo-dataset).
- [Current AImoclips statistical analysis repository](https://github.com/HunRotation/aimoclips-analysis).

MTG-Jamendo metadata retains its upstream CC BY-NC-SA 4.0 license, distinct
from the AImoclips dataset license. Reference audio is third-party material
with individual licensing terms. The package shares identifiers, metadata
and checksums with attribution, not a relicensed copy of the audio.
