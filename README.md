# Wavelet analysis of inflation expectations and consumer behavior
This repository is the underlying code for my paper: [Expectations in Time and Frequency](https://drive.google.com/file/d/19PrPUhgOJNLO4LCoTa9-BGYzR9Ebqd02/view).

## Setting up the environment
To create a virtual environment using `conda`, use:
```
conda env create --name inflation --file environment.yml
```
To create a virtual environment using `pip`, use:
```
pip install -r requirements.txt
```

Once, you have created the environment, install the internal packages using:
```
pip install -e .
```

Finally, run the environment. For a `conda` virtual environment, use:
```
conda activate inflation
```
For a `venv` generated via `pip`, use:
```
venv\Scripts\activate.bat
```

## API access
You will need to create personal accounts on the different database's websites and request API credentials. Save them to a file named `.env` in a root folder.

See `.example_env` for how you should store the credentials.

<b>Make sure that the file name is exactly `.env` (nothing before the `.` nor after `env`) so that your credentials are not exposed in your git repository!</b>

## Results
To generate the results, simply run the `results.py` file. It can also be run in an interactive terminal in VSCode, similar to a Jupyter Notebook.

## Simulated data
See scripts files for some simple simulations with simulated cyclical income and consumption functions from:
Ramsey, J. B., Gallegati, M., Gallegati, M., & Semmler, W. (2010). Instrumental variables and wavelet decompositions. Economic Modelling, 27(6), 1498–1513. https://doi.org/10.1016/j.econmod.2010.07.011

Project structure inspired by [The Good Research Code Handbook](https://goodresearch.dev/#the-good-research-code-handbook) by Patrick Mineault.