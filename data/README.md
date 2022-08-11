# Training and Validation Data

[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](https://github.com/RikGhosh487/Metallicity-Approximator/blob/main/LICENSE) ![Format](https://img.shields.io/badge/format-.csv-rgb(12%2C%2093%2C%20148))

This **directory** contains two subdirectories, each of which contains a CSV file obtained from the SQL [query](https://github.com/RikGhosh487/Metallicity-Approximator/blob/main/data_extraction.sql). The entire data (train and validation combined) can be found in `segue.csv`.

The train dataset has a size of 111644 individual observations. This dataset contains values from the **5** SDSS PSF filters and the corresponding spectroscopic metallicity for the star.

The valid dataset has a size of 27911 individual observations. This dataset contains values from the **5** SDSS PSF filters and the corresponding spectroscopic metallicity for the star.

The actual contents of the two datasets can be randomly shuffled by running the `data_shuffle.py` script. After navigating to the appropriate directory, run the following command in your shell:
```
>>> python data_shuffle.py
```
you can alter the sizes of the training and validation data, by modifying the `TEST_RATIO` parameter in the [file](https://github.com/RikGhosh487/Metallicity-Approximator/blob/main/data/data_shuffle.py).