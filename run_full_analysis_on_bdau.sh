#!/bin/bash

alias pipenv='/opt/python/3.8.6/bin/pipenv'

pipenv run taskset -c 32-39 python 01_train_test_split.py
pipenv run taskset -c 32-39 python 02_train_eval_current_model.py
pipenv run taskset -c 32-39 python 03_plot_current_model.py
pipenv run taskset -c 32-39 python 04_consolidate_indications.py
pipenv run taskset -c 32-39 python 05_wrangling_and_mice.py
#pipenv run taskset -c 32-39 python 06_categorical_imputation.py
#pipenv run taskset -c 32-39 python 07_albumin_imputation.py
#pipenv run taskset -c 32-39 python 08_lactate_imputation.py
#pipenv run taskset -c 32-39 python 09_train_eval_novel_model.py
