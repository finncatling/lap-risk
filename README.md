# lap-risk

This analysis code relates to the study *Highlighting uncertainty in clinical risk prediction: A model of mortality risk in emergency laparotomy developed using data from 186 hospitals*.

![Tests](https://github.com/finncatling/lap-risk/workflows/Tests/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/finncatling/lap-risk/badge.svg?t=H4at4E)](https://coveralls.io/github/finncatling/lap-risk)


## Study data

The NELA dataset is required to run the full analysis, and is not included in this repository. Under the terms of the data sharing agreement for this study we are unable to share the source data directly. Requests for anonymous patient-level data can be made directly to the NELA Project Team. 

Without these data, it is still possible to set up the project environment and run the tests.


## Environmental setup

This codebase is written in Python 3.8.6 and R 4.0.3. We manage dependencies with pipenv and renv. To set up your project environment, you'll need to:

- Install Python and pip
- [Install pipenv](https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv)
- Install R and renv

If working in the Imperial BDAU, pipenv isn't in path. We can define a convenient alias for it in this case:

```console
alias pipenv='/opt/python/3.8.6/bin/pipenv'
```

Then install dependencies with  `pipenv install` and `renv::restore()`


## Running the analysis

The shell script `run_full_analysis_on_bdau.sh` runs the full analysis (not the initial enironmental setup) on the Imperial BDAU. It runs all scripts in sequence inside our pipenv and renv environments, restricting itself to 8 cores. 


### Running the analysis scripts individually

Enter the project's python environment:

```console
pipenv shell
```

Then run the python scripts in the root directory in numerical order, i.e. starting with:

```console
python 00_initial_data_wrangling.py
```


## Running the tests

Install development packages with:

```console
pipenv install --dev 
```

Run tests for the `utils` module and check coverage:

```console
pytest --cov=utils tests/ 
```