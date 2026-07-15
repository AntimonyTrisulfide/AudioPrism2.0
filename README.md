# AudioPrism 2.0

AudioPrism 2.0 is a research-oriented music source separation system built to
replace the magnitude-mask U-Net used in AudioMask 1.x. It separates a mono mix
into the dataset's instrument classes with a source-conditioned, band-split
axial transformer that predicts complex ratio masks.

The project is designed as a self-contained HPC transfer bundle. It reads the
existing phase-aware AudioMask `.pt` datasets directly, supports single- and
multi-GPU training, resumes interrupted PBS jobs with full optimizer state, evaluates per-source
SI-SDR/SNR, and separates arbitrary audio with overlap-add inference.

## Why version 2 exists

The strongest 1.x validation checkpoint reached about -4.98 dB SI-SDR, while
the same model with oracle source phase was materially better. Training loss
continued falling after validation SI-SDR peaked, indicating both a
reconstruction ceiling and objective/metric mismatch. Version 2 addresses
those failure modes directly:

- Complex masks estimate real and imaginary STFT components instead of reusing
  mixture phase.
- Mel-spaced band splitting gives low and high frequencies different learned
  projections without processing all 1025 bins as independent tokens.
- Alternating temporal and cross-band attention models long events and harmonic
  structure.
- Learned source embeddings share one separator backbone across all instrument
  classes while preserving source-specific decoding.
- An activity head handles absent instruments explicitly.
- A differentiable mixture-consistency projection ensures that estimated stems
  sum to the input mixture.
- The objective combines complex reconstruction, log magnitude, spectral
  convergence, waveform L1, SI-SDR, activity classification, and consistency.

The architecture is informed by
[BS-RoFormer](https://arxiv.org/abs/2309.02612),
[Mel-RoFormer](https://arxiv.org/abs/2310.01809), and
[TF-GridNet](https://arxiv.org/abs/2211.12433), while remaining a compact,
dependency-light implementation suited to the available dataset and cluster.

## Repository layout

```text
AudioPrism2/
|-- audioprism/           model, data contract, losses, metrics, engine
|-- configs/              reproducible base and local smoke configurations
|-- scripts/              PBS submission and environment bootstrap
|-- tests/                model invariants and data-transform tests
|-- train.py              AMP/DDP training and exact resume
|-- evaluate.py           checkpoint evaluation with per-source metrics
|-- separate.py           overlap-add inference for arbitrary audio
|-- oracle_ceiling.py     IRM, mixture-phase, and identity ceilings
|-- inspect_dataset.py    fast compatibility and metadata audit
`-- plot_metrics.py       portfolio-ready learning curves
```

## Data contract

No HPC dataset conversion is required if it was produced by the phase-aware
AudioMask preprocessor. A dataset directory must contain:

```text
preprocessed_train/
|-- dataset_metadata.pt
|-- Track00001/
|   |-- metadata.pt
|   |-- mix.pt
|   |-- Vocal.pt
|   |-- Guitar.pt
|   `-- ...
`-- Track00002/
```

Each source payload must contain `spectrogram` and either `true_phases` or
`phases`. `dataset_metadata.pt` defines the authoritative source order and the
`sr`, `n_fft`, `hop_length`, and `chunk_samples` values. The base configuration
matches the last 8-second run: 16 kHz, FFT 2048, hop 512, and 128,000 samples.

Audit the cluster data before training:

```bash
python inspect_dataset.py /path/to/preprocessed_train --scan-chunks 16
python oracle_ceiling.py \
  --data-dir /path/to/preprocessed_val \
  --output oracle_validation.json
```

The oracle report is an important experiment gate. It reveals how much of the
remaining error is caused by phase, magnitude masking, source labels, or metric
implementation before a long run starts.

## Local verification

From inside `AudioPrism2`:

```bash
python -m pip install -r requirements-hpc.txt
python -m unittest discover -s tests -v
python inspect_dataset.py ../tiny_preprocessed_train --scan-chunks 2 \
  --allow-mixture-phase-targets
python train.py --config configs/smoke.yaml
```

The local fixture predates source-phase storage, so the smoke config explicitly
uses mixture phase for its targets. This fallback is disabled in `base.yaml`
and must not be used for a real 2.0 experiment. The smoke configuration proves that data loading,
forward/backward passes, validation, and checkpoint serialization work; it is
not a meaningful separation experiment.

## HPC training

Copy this directory to the cluster as one unit. The dataset and run outputs
should remain outside the code directory when possible.

Create an environment once:

```bash
cd AudioPrism2
ENV_NAME=audioprism2 bash scripts/bootstrap_hpc.sh
```

By default, the PBS job preprocesses train and validation data into its
job-local `$TMPDIR` and removes that large intermediate data automatically when
the job ends. Raw audio, split manifests, checkpoints, and logs remain in home.
This matches clusters that expose `/var/tmp/pbs.*` scratch only on compute
nodes. Submit the base experiment:

```bash
export OUTPUT_DIR=/cluster/path/audioprism2_runs
export PYTHON_BIN=$HOME/.conda/envs/audioprism2/bin/python
export RUN_NAME=bsct_16k_seed42
export PREPROCESS_PROJECT=$HOME/AudioPrism_HPC
export PREPROCESS_IN_JOB=1
bash scripts/submit.sh
```

`scripts/train.pbs` requests the existing `dgx` queue with one GPU, 12 CPUs,
64 GB RAM, and 24 hours. Adjust only the `#PBS` resource lines if the site's
limits differ. The launcher automatically resumes
`runs/<name>/checkpoints/latest.pt` when it exists. For a multi-GPU allocation,
change `ngpus` in the PBS resource line and export `NGPUS` to the same value.
To reuse persistent preprocessed data instead, set `PREPROCESS_IN_JOB=0` and
export `TRAIN_DIR` and `VAL_DIR`; submission validates both metadata files
before requesting a GPU.

Any config value can be overridden without editing YAML:

```bash
python train.py --config configs/base.yaml \
  --set data.train_dir=/data/train \
  --set data.val_dir=/data/val \
  --set train.batch_size=4 \
  --set name=bsct_batch4
```

For GPU memory pressure, reduce `train.batch_size` first and increase
`train.accumulation_steps` to preserve effective batch size. Then reduce
`model.dim` from 192 to 160. Keep the number of bands and chunk duration fixed
for the first controlled comparison.

## Evaluation and inference

Evaluate the best checkpoint on held-out test data:

```bash
python evaluate.py \
  --checkpoint runs/bsct_16k_seed42/checkpoints/best.pt \
  --data-dir /cluster/path/preprocessed_test \
  --output runs/bsct_16k_seed42/test_metrics.json
```

Separate a full song:

```bash
python separate.py \
  --checkpoint runs/bsct_16k_seed42/checkpoints/best.pt \
  --input song.wav \
  --output-dir outputs/song
```

Generate learning curves:

```bash
python plot_metrics.py \
  runs/bsct_16k_seed42/logs/metrics.jsonl \
  --output runs/bsct_16k_seed42/training_curves.png
```

## Experiment protocol

For a portfolio or report, keep the first comparison controlled:

1. Use the exact train/validation/test split from AudioMask 1.x.
2. Keep 16 kHz, 8-second chunks, FFT 2048, and hop 512.
3. Run `oracle_ceiling.py` and retain its JSON output.
4. Train three seeds (`42`, `123`, `456`) with identical hyperparameters.
5. Select checkpoints only by validation SI-SDR.
6. Report test SI-SDR and SNR globally and per active source.
7. Include parameter count, training hardware, wall-clock time, and failure
   cases; do not compare the best test checkpoint chosen on test performance.

Suggested ablations are complex masks versus mixture phase, axial attention
versus temporal-only attention, activity loss on/off, and mixture consistency
on/off. Change one factor at a time.

## Outputs and reproducibility

Every run stores the resolved `config.yaml`, append-only `metrics.jsonl`, text
logs, `latest.pt`, SI-SDR-selected `best.pt`, and periodic epoch checkpoints.
A checkpoint contains model, optimizer, scheduler, AMP scaler, epoch, best
metric, source ordering, and full configuration, so evaluation does not rely on
remembering command-line settings.

AudioPrism 2.0 is research software. Results depend on stem quality, class
mapping, silence policy, and dataset split; the included oracle and per-source
reports are meant to make those assumptions visible rather than hide them.
