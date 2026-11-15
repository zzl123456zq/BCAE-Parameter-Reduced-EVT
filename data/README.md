# Data directory

No dataset is distributed with this code.

The public Tennessee Eastman Process (TEP) reference data can be downloaded from:

- Repository: https://github.com/camaramm/tennessee-eastman-profBraatz
- ZIP archive: https://github.com/camaramm/tennessee-eastman-profBraatz/archive/refs/heads/master.zip

To rerun the data-dependent notebook cells, obtain the datasets through their authorized sources and provide a local preprocess.py module exposing:

- ld_dataset(construct_long_short=False, imb=0, known_index=..., ...)
- tep_dataset(train_index=..., test_index=...)

Then set the BCAE_DATASET_MODULE_DIR environment variable to the directory containing preprocess.py. See the repository README for an example.

Do not commit restricted raw data, processed arrays, or private filesystem paths to this directory.
