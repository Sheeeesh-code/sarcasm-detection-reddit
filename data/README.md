# Data

The dataset is not stored in git (the files are too large and it is third-party data).

1. Download the **balanced** split of SARC (Khodak et al., 2018) and place these two files here:
   - `train-balanced.csv.bz2`
   - `test-balanced.csv.bz2`
2. Run `../data_processing.ipynb`. It produces:
   - `train_cleaned.csv`
   - `test_cleaned.csv`

All other notebooks read the two cleaned files.
