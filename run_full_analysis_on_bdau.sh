#!/bin/bash

shopt -s expand_aliases
alias pipenv='/opt/python/3.8.6/bin/pipenv'

pipenv run taskset -c 32-39 python 00_initial_data_wrangling.py
pipenv run taskset -c 32-39 python 01_train_test_split.py
pipenv run taskset -c 32-39 python 02_0_export_current_model_data_to_r.py
taskset -c 32-39 Rscript 02_1_train_current_model_and_predict.R
pipenv run taskset -c 32-39 python 02_2_eval_current_model_predictions.py
pipenv run taskset -c 32-39 python 03_plot_current_model.py
pipenv run taskset -c 32-39 python 04_consolidate_indications.py
pipenv run taskset -c 32-39 python 05_wrangling_and_mice.py
pipenv run taskset -c 32-39 python 06_categorical_imputation.py
pipenv run taskset -c 32-39 python 07_albumin_lactate_imputation.py
pipenv run taskset -c 32-39 python 08_0_train_eval_novel_model.py
pipenv run taskset -c 32-39 python 08_1_eval_novel_model_distributions.py
pipenv run taskset -c 32-39 python 09_plot_novel_model.py
pipenv run taskset -c 32-39 python 10_compare_current_and_novel_models.py
pipenv run taskset -c 32-39 python 11_albumin_lactate_imputation_mortality_sensitivity.py
pipenv run taskset -c 32-39 python 12_generate_demographic_table.py
pipenv run taskset -c 32-39 python 13_train_production_models.py
