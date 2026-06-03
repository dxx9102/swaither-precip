# Container image

This directory contains the Singularity definition for a reproducible
SwAIther-Precip container image. **This is the recommended installation
path for HPC clusters.**

## File

- `swaither_precip.def` — Singularity / Apptainer definition file, based
  on `nvcr.io/nvidia/pytorch:25.08-py3`. Supports both x86_64 and aarch64
  architectures (including NVIDIA GH200).

## What's in the image

- NVIDIA's official PyTorch base image (`pytorch:25.08-py3`) with CUDA-enabled
  PyTorch already installed
- A micromamba `swaither-precip` environment on top with all SwAIther-Precip
  dependencies (Hydra, xarray, cartopy, rioxarray, wandb, xskillscore,
  properscoring, etc.)
- `nvidia-physicsnemo==1.2.0` (the tested version) installed via pip with
  `--no-deps` to avoid pulling in a second PyTorch
- A site-packages bridge that exposes the base image's PyTorch to the
  micromamba env (see comments in the `.def` file for the gory details)

## Building

The recipe builds in two ways depending on your privileges:

### With root (e.g. on a personal machine)

```bash
sudo singularity build swaither_precip.sif containers/swaither_precip.def
```

### Without root (HPC login node)

```bash
singularity build --fakeroot swaither_precip.sif containers/swaither_precip.def
```

Most modern HPC clusters support `--fakeroot`. If yours doesn't, use the
`--remote` option (builds on Sylabs Cloud and downloads the result):

```bash
singularity build --remote swaither_precip.sif containers/swaither_precip.def
```

Build time is roughly 15–30 minutes depending on the base-image download
speed and your CPU.

## Using the image

### Interactive shell

```bash
export SINGULARITY_BINDPATH="/path/to/data,/path/to/checkpoints"
repo_path=/path/to/swaither-precip   # absolute path to your clone

singularity shell --nv \
    -B "${repo_path}:/ws" \
    -W /ws \
    swaither_precip.sif

# Inside the container, the env is activated automatically; verify:
python -c "import physicsnemo, torch; print(physicsnemo.__version__, torch.__version__)"
```

### Running a single command

The `%runscript` block activates the env and runs whatever command you pass:

```bash
singularity exec --nv \
    -B "${repo_path}:/ws" \
    -W /ws \
    swaither_precip.sif \
    python scripts/train_weekly.py \
        --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml
```

### Via SLURM

A minimal SLURM submission script:

```bash
#!/bin/bash -l
#SBATCH --account YOUR_ACCOUNT
#SBATCH --job-name swaither_train
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200G
#SBATCH --time 24:00:00

export SINGULARITY_BINDPATH="/path/to/data,/path/to/repo"
container=/path/to/swaither_precip.sif
repo_path=/path/to/swaither-precip

singularity exec --nv -B "${repo_path}:/ws" -W /ws "$container" \
    micromamba run -p /opt/micromamba/envs/swaither-precip \
    python -u scripts/train_weekly.py \
        --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml
```

## Why use the container instead of a local conda install?

- The base image already has PyTorch + CUDA properly set up for NVIDIA GPUs,
  including support for NVIDIA GH200 (aarch64).
- All system libraries (GEOS, PROJ, GDAL, ...) are pre-installed.
- The `nvidia-physicsnemo==1.2.0` pin matches the version SwAIther-Precip
  was developed against — no risk of incompatibility from upstream changes.
- Reproducibility: the image is a fixed snapshot of the environment used to
  produce the paper's results.
- Singularity works on virtually every HPC cluster, while Docker often does not.

The downsides are that the image is large (~10 GB) and requires Singularity
to build. For laptops and dev workstations, the standalone conda env
(`environment.yml` in the repo root) is a fine alternative.

## A note for Docker users

If you prefer Docker, you can read this `.def` file as a recipe and adapt it
into a Dockerfile — the underlying build steps are the same. The base image
(`nvcr.io/nvidia/pytorch:25.08-py3`) is pulled from NVIDIA NGC and works
identically in either ecosystem.
