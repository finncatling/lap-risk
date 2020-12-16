# lap-risk

Modelling mortality risk in emergency laparotomy, using data from the NELA.

![Tests](https://github.com/finncatling/lap-risk/workflows/Tests/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/finncatling/lap-risk/badge.svg?t=H4at4E)](https://coveralls.io/github/finncatling/lap-risk)


## Install

This codebase is written in Python 3.8.6 and R 4.0.3. We manage dependencies with pipenv and renv. To get started, you'll need to:

- Install Python and pip
- [Install pipenv](https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv)
- Install R and renv

If working in the Imperial BDAU, pipenv isn't in path. We can define a convenient alias for it in this case:

```console
alias pipenv='/opt/python/3.8.6/bin/pipenv'
```

Then install dependencies with  `pipenv install` and `renv::restore()`


## Running the analysis

Enter the project's python environment:

```console
pipenv shell
```

Then run the python scripts in the root directory in numerical order, i.e. starting with:

```console
python 00_initial_data_wrangling.py
```

### Limiting the number of cores used

When running in the Imperial BDAU, we should limit the number of cores used for the computationally-intensive bits of the analysis. E.g. to run on the first 8 cores:

```console
taskset -c 0-7 python 00_initial_data_wrangling.py
```

The convenience shell script `run_full_analysis_on_bdau.sh` automates running the analysis (not the initial install) on the BDAU. It runs all scripts in sequence inside our pipenv and renv environments, restricting itself to 8 cores. 


## Running tests

Install dev packages with:

```console
pipenv install --dev 
```

Run tests and check coverage:

```console
pytest --cov=utils tests/ 
```

NB. Tests are currently only written for the `utils` module as this provides the functions upon which we build the analysis