# lap-risk

Modelling mortality risk in emergency laparotomy, using data from the NELA

## Running on the Imperial BDAU

Set the correct python environment by running the following in the terminal:

```console
alias python3=/opt/python/3.6.8/bin/python3.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/python/3.6.8/lib
```

Then, after navigating to the `lap-risk` root directory, you _could_ run the individual analysis files as follows:

```console
python3 1_train_test_split.py
```

### Limiting the number of cores used

We want to use `taskset` to limit the number of cores used for the computationally-intensive bits of the analysis, but the `python3` alias set above doesn't work with it. So, we need to instead use the full python path instead. E.g. to run on the first 16 cores:

```console
taskset -c 0-15 /opt/python/3.6.8/bin/python3.6 1_train_test_split.py
```