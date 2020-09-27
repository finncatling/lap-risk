# lap-risk

Modelling mortality risk in emergency laparotomy, using data from the NELA

## Install

Install necessary dependencies with [pipenv](https://pipenv-fork.readthedocs.io/en/latest/):

```console
pipenv install
```

## Running the analysis

Enter the project's python environment:

```console
pipenv shell
```

Then run the python scripts in the root directory in numerical order, e.g. starting with:

```console
python 01_train_test_split.py
```

### Limiting the number of cores used

When running in the Imperial BDAU, we should limit the number of cores used for the computationally-intensive bits of the analysis. E.g. to run on the first 16 cores:

```console
taskset -c 0-15 python 01_train_test_split.py
```

## Running tests
Install dev packages with:

```console
pipenv install --dev 
```

Run tests and check coverage:

```console
pytest --cov=utils tests/ 
```

NB. Tests are currently only written for the utils module as this provides the functions upon which we build the analysis