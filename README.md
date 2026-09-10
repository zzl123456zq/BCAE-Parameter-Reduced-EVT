# BCAE With Parameter-Reduced EVT

This directory contains the implementation used to train and evaluate the proposed bidirectional contrastive reconstruction model and its EVT-based decision rule.

## Release scope

The release includes the model, bidirectional contrastive objective, EVT threshold estimator, pipeline and TEP experiments, computational-efficiency utilities, a data-loader adapter, the contamination utility, environment requirements, and machine-readable copies of the reported results.

The industrial pipeline data cannot be redistributed because of data-use restrictions. Consequently, main.ipynb retains the executed outputs as an audit trail, while rerunning its data-dependent cells requires an authorized local copy of the dataset and preprocessing module. No raw industrial data, trained weights, or author-specific filesystem paths are included.

## Files

- main.ipynb: pipeline experiment, computational-efficiency evaluation, TEP validation, and EVT sensitivity analysis, with preserved outputs.
- model.py: BCAE architecture and bidirectional contrastive loss.
- evt.py: parameter-reduced EVT threshold estimation.
- computational_complexity.py: parameter-count and inference-latency utilities.
- data_adapter.py: documented interface to an authorized external preprocessing module.
- contamination.py: training-set contamination generator used in the robustness study.
- verify_installation.py: data-free smoke test for the model, loss, and EVT code.
- verify_reported_results.py: consistency check for the released tables and saved notebook outputs.
- results/: machine-readable reported metrics.

## Environment

The notebook metadata records Python 3.9.18. Create a clean environment and install:

    python -m pip install -r requirements.txt
    python verify_installation.py
    python verify_reported_results.py

The released code was smoke-tested in the recorded notebook environment: Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.6.1, Matplotlib 3.3.0, and PyTorch 2.6.0+cu126. The ranges in requirements.txt permit compatible installations on machines with different CUDA configurations.

## Connecting an authorized dataset

The external preprocessing module must be named preprocess.py and provide the functions ld_dataset and tep_dataset. Set BCAE_DATASET_MODULE_DIR to the directory containing that file before starting Jupyter.

The Tennessee Eastman Process (TEP) reference data used in the cross-dataset experiment are publicly available from the [Prof. Braatz TEP repository](https://github.com/camaramm/tennessee-eastman-profBraatz). The complete dataset can be obtained through [Download ZIP](https://github.com/camaramm/tennessee-eastman-profBraatz/archive/refs/heads/master.zip) or Git:

    git clone https://github.com/camaramm/tennessee-eastman-profBraatz.git

This repository contains the training files d00.dat--d21.dat and the corresponding test files d00_te.dat--d21_te.dat used by the TEP preprocessing routine.

PowerShell example:

    $env:BCAE_DATASET_MODULE_DIR = "D:\authorized_dataset_module"
    jupyter notebook main.ipynb

The adapter raises a descriptive error if the module is unavailable. This keeps private paths and restricted data out of the public release.

## Experimental protocol

- Random seed: 42 unless otherwise stated in a reported experiment.
- Training: only known-condition samples are used and class labels are not used by BCAE optimization.
- Detection score: sample-wise mean squared reconstruction error.
- Decision rule: the EVT module estimates the reconstruction-error threshold from the training errors.
- Dataset labels are used only to construct the predefined experimental split and to calculate evaluation metrics; they are not supplied to BCAE during training.

Saved notebook outputs are provided for inspection. The CSV files in results summarize the values reported in the manuscript and response letter; they are not recomputed when the files are opened. Run verify_reported_results.py to check these files and the saved deterministic notebook metrics against the paper. Training time, inference latency, and GPU memory are hardware- and run-dependent; results/computational_efficiency.csv records the rounded reference-run values used in the paper, while the notebook retains its actual measured values and labels the paper reference separately.

## Compared methods and official sources

The baseline implementations were developed or adapted with reference to the following author-maintained repositories when official code was available. For methods whose authors did not provide a public implementation, the formal publication page is listed instead. Because the original repositories target different data modalities and experimental settings, their architectures and input interfaces were adapted to the common multivariate time-series protocol described in the manuscript.

- VAE: [official IEEE publication](https://doi.org/10.1109/TASE.2020.3035620); a maintained implementation is available in the [PyOD outlier-detection library](https://github.com/yzhao062/pyod/blob/master/pyod/models/vae.py).
- DAGMM: [official ICLR paper](https://openreview.net/forum?id=BJJLHbb0-); a benchmarked implementation is available in the [ADBench anomaly-detection library](https://github.com/Minqi824/ADBench/tree/main/adbench/baseline/DAGMM).
- GANomaly: [authors' official implementation](https://github.com/samet-akcay/ganomaly).
- USAD: [authors' official implementation](https://github.com/manigalati/usad).
- OpenMax: [authors' official OSDN implementation](https://github.com/abhijitbendale/OSDN).
- OSSC+EVT: [official IEEE publication](https://doi.org/10.1109/TII.2022.3169459) (no author-maintained public code was identified).
- D3R: [authors' official implementation](https://github.com/ForestsKing/D3R).
- GEL: [authors' official implementation](https://github.com/00why00/Glocal).
- TCDL: [official publisher page](https://doi.org/10.1016/j.knosys.2024.112932) (no author-maintained public code was identified).

## Data and license notes

Data availability is distinct from code availability. Users must obtain the industrial datasets under their respective terms. A software license is intentionally not asserted in this directory; the repository owner should add the intended license before public distribution.
