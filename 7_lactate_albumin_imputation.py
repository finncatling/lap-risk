#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8

# # Imputation models for lactate and albumin

# These should be usable in production, so they shouldn't use variables that are not used in the final mortality risk GAM.
# 
# **TODO:** remove variables that are unused in the final mortality risk GAM

# In[1]:


import os, copy, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pygam
from progressbar import progressbar as pb
from pygam import GAM, LinearGAM, GammaGAM, s, f, l, te
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (r2_score,
                             mean_squared_error, median_absolute_error,
                             mean_absolute_error, mean_gamma_deviance)
from typing import Dict
                             

sys.path.append('')
from nelarisk.constants import RANDOM_SEED
from nelarisk.helpers import (load_object, save_object, GammaTransformer,
                              check_system_resources)
from nelarisk.impute import ContImpPreprocessor
from nelarisk.gam import combine_mi_gams, quick_sample
from nelarisk.evaluate import evaluate_samples


check_system_resources()


# In[2]:


op = load_object(os.path.join('data', 'imputation_output.pkl'))


# ## Helper functions

# In[3]:


def inspect_y_trans(d: Dict,
                    hist_args = {'bins': 20, 'alpha': 0.5}):
    """Compare original y (albumin or lactate) and its transformation."""
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.ravel()

    for i, fold in enumerate(('train', 'test')):
        ax[i].hist(d['pp'][fold].y_train,
                   label='original', **hist_args)
        ax[i].hist(d['pp'][fold].y_train_trans,
                   label='transformed', **hist_args)
        ax[i].set(title=fold)
        ax[i].legend()

    plt.tight_layout()
    plt.show()


# In[4]:


def compare_folds(d: Dict):
    """Sample new values for imputation model and compare to test set.
    
        TODO: Evaluate performance across all imputed DataFrames,
        not just 0"""
    for fold in ['train', 'test']:
        print(fold.upper())

        print('Evaluating point estimates for transformed y...')
        X0, y = d['pp'][fold].yield_train_X_y(0)
        _, y_trans = d['pp'][fold].yield_train_X_y(0, trans_y=True)
        y0_pred = d['gam'].predict(X0)

        for metric in (r2_score, mean_squared_error,
                       median_absolute_error, mean_absolute_error):
            print(f'{metric.__doc__.splitlines()[0]} =',
                  f'{metric(y_trans, y0_pred)}')

        plt.hist(y_trans - y0_pred, bins=50)
        plt.xlabel('Residuals')
        plt.show()

        print('Sampling y then evaluating samples...')
        samples = quick_sample(d['gam'], X0, RANDOM_SEED, n_draws=1000)
        inv_samples = d['trans'].inverse_transform(
            samples.flatten().reshape(-1, 1))
        inv_samples = inv_samples.reshape(samples.shape)

        evaluate_samples(y, inv_samples)


# ## Lactate imputation GAM

# In[5]:


lac = {'pp': {},
       'trans': QuantileTransformer(output_distribution='normal',
                                    random_state=RANDOM_SEED)}


# Instantiate and run preprocessors
for fold in ('train', 'test'):
    lac['pp'][fold] = ContImpPreprocessor(
        op[fold]['imp'],
        target='S03PreOpArterialBloodLactate',
        drop_vars=['S03PreOpLowestAlbumin',
                   'S03WhatIsTheOperativeSeverity',
                   'S03CardiacSigns',
                   'S03RespiratorySigns',
                   'S03DiagnosedMalignancy',
                   'S03Pred_Peritsoil',
                   'S03Sodium'] + op['missingness_vars'])
    lac['pp'][fold].preprocess()

    
# Transform lactate to Gaussian
lac['pp']['train'].y_train_trans = lac['trans'].fit_transform(
    lac['pp']['train'].y_train.reshape(-1, 1)).flatten()
lac['pp']['test'].y_train_trans = lac['trans'].transform(
    lac['pp']['test'].y_train.reshape(-1, 1)).flatten()

inspect_y_trans(lac)


# Check that at least one of each indication has been imputed in all imputed DataFrames (this will cause problems with parameter dimensionality mismatch when combining GAMs if it is not the case).

# In[6]:


n_inds = lac['pp']['train'].imp_dfs[0]['Indication'].nunique()

for i in range(lac['pp']['train'].n_imp_dfs):
    assert lac['pp']['train'].imp_dfs[i]['Indication'].nunique() == n_inds


# In[7]:


for i, c in enumerate(lac['pp']['train'].X['train'][0].columns):
    print(i, c)


# Fit GAMs on each imputed DataFrame and combine to a single GAM.

# In[8]:


lac_gams = []

for i in pb(range(lac['pp']['train'].n_imp_dfs), prefix='GAM fit'):
# for i in range(2):  # for quick training during model development
    lac_gams.append(LinearGAM(
        s(0, lam=480) +  #age
        s(2, lam=550) +  #k
        s(4, lam=600) +  #wcc
        s(6, lam=500) +  #sbp
        s(7, lam=600) +  #gcs
        f(10, coding='dummy') +  #asa
        f(11, coding='dummy') +  #urgency
        te(12, 8,  # indication & ct
           lam=(2, 0.2),
           n_splines=(len(op['multi_cat_levels']['Indication']), 2),
           spline_order=(0, 0),
           dtype=('categorical', 'categorical')) +
        te(5, 9,  # pulse & ecg
           lam=(200, 2),
           n_splines=(20, 2),
           spline_order=(3, 0),
           dtype=('numerical', 'categorical')) +
        te(1, 3, lam=18.0)  #creat & urea
    ).fit(*lac['pp']['train'].yield_train_X_y(i, trans_y=True)))

lac['gam'] = combine_mi_gams(lac_gams)


# Visualise partial dependence for each GAM feature:

# In[9]:


terms = lac['gam'].terms.info['terms'][:-1]

n_cis = 5
cis = np.linspace(0.025, 0.975, n_cis * 2)

n_rows = int(np.ceil((len(terms) - 1) / 2))
fig, ax = plt.subplots(n_rows, 2, figsize=(8, 2.5 * n_rows))
ax = ax.ravel()

subplot_labels = {7: {'lines': ['No CT', 'CT'],
                      'title': 'Indication'},
                  8: {'lines': ['Normal ECG', 'Arrhythmia'],
                      'title': 'Heart rate'}}

for i, term in enumerate(terms):
    if term['term_type'] != 'tensor_term':
        XX = lac['gam'].generate_X_grid(term=i)
        pdep, confi = lac['gam'].partial_dependence(term=i,
                                                    quantiles=cis)
        
        j = term['feature']
        
        for k in range(n_cis):
            ax[i].fill_between(XX[:, j], confi[:, k], confi[:, -(k + 1)],
                               alpha=1/n_cis, color='black', lw=0.0)
        
        ax[i].set_title(lac['pp']['train'].X['train'][0].columns[j])
    
    else:
        XX = lac['gam'].generate_X_grid(term=i, meshgrid=True)
        Z, confi = lac['gam'].partial_dependence(term=i, X=XX,
                                                 quantiles=cis,
                                                 meshgrid=True)
        if i in (7, 8):
            colours = ['blue', 'red']
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax[i].fill_between(XX[0][:, 0], confi[:, sli, k],
                                       confi[:, sli, -(k + 1)], lw=0.0,
                                       alpha=1/n_cis, color=colours[l])
                lines.append(Line2D([0], [0], color=colours[l]))

            ax[i].legend(lines, subplot_labels[i]['lines'],
                         loc='upper left')
            ax[i].set_title(subplot_labels[i]['title'])
        
        else:
            # Finish plotting other PD subplots before starting 3D plot
            plt.tight_layout()
            plt.show()
            
            # Make separate 3d plots for cont-cont tensor interactions
            ax3d = plt.axes(projection='3d')
            ax3d.plot_surface(XX[0], XX[1], Z, cmap='viridis')
            ax3d.view_init(30, 150)
            ax3d.set_xlabel(lac['pp']['train'].X['train'][0].columns[
                term['terms'][0]['feature']])
            ax3d.set_ylabel(lac['pp']['train'].X['train'][0].columns[
                term['terms'][1]['feature']])
            plt.show()


# Make a more-informative plot for the indications feature:

# In[10]:


i = 7
colours = ['blue', 'red']
lines = []

j = lac['gam'].terms.info['terms'][i]['terms'][0]['feature']

XX = lac['gam'].generate_X_grid(
    term=i, n=len(op['multi_cat_levels']['Indication']), meshgrid=True)
Z, confi = lac['gam'].partial_dependence(term=i, X=XX,
                                         quantiles=cis,
                                         meshgrid=True)

fig, ax = plt.subplots(figsize=(9, 4))

for l, sli in enumerate([0, -1]):
    for k in range(n_cis):
        ax.fill_between(XX[0][:, 0], confi[:, sli, k],
                        confi[:, sli, -(k + 1)], lw=0.0,
                        alpha=1/n_cis, color=colours[l])
    lines.append(Line2D([0], [0], color=colours[l]))

ax.legend(lines, subplot_labels[i]['lines'], loc='upper left')
ax.set_title(subplot_labels[i]['title'])
ax.set_xticks(XX[0][:, j])
ax.set_xticklabels(op['multi_cat_levels']['Indication'],
                   rotation=45, rotation_mode='anchor',
                   horizontalalignment='right',
                   verticalalignment='top')
ax.set_xlim([XX[0][0, j], XX[0][-1, j]])
plt.grid(linewidth=0.5, linestyle=':')

plt.show()


# ### Sample new values and compare to test set

# In[11]:


compare_folds(lac)


# **TODO:** Investigate ways to improve the lactate imputation model to improve upon the lower left plot above

# ## Albumin

# In[12]:


alb = {'pp': {},
       'trans': GammaTransformer(
           op['winsor_thresholds']['S03PreOpLowestAlbumin'])}


# Instantiate and run preprocessors
for fold in ('train', 'test'):
    alb['pp'][fold] = ContImpPreprocessor(
        op[fold]['imp'],
        target='S03PreOpLowestAlbumin',
        drop_vars=['S03PreOpArterialBloodLactate'] + op['missingness_vars'])
    alb['pp'][fold].preprocess()
    
    # Transform albumin to (approximate) Gamma
    alb['pp'][fold].y_train_trans = alb['trans'].transform(
        alb['pp'][fold].y_train)

inspect_y_trans(alb)


# In[13]:


for i, c in enumerate(alb['pp']['train'].X['train'][0].columns):
    print(i, c)


# Fit GAMs on each imputed DataFrame and combine to a single GAM.

# In[14]:


alb_gams = []

for i in pb(range(alb['pp']['train'].n_imp_dfs), prefix='GAM fit'):
# for i in range(2):  # for quick training during model development
    alb_gams.append(GammaGAM(
        s(0, lam=500) +  #age
        s(1, lam=300) +  #creat
        s(2, lam=400) +  #na
        s(3, lam=300) +  #k
        s(4, lam=400) +  #urea
        s(5, lam=400) +  #wcc
        s(7, lam=400) +  #sbp
        s(8, lam=300) +  #gcs
        f(12, coding='dummy') +  #asa
        f(16, coding='dummy') +  #perit soil
        f(17, coding='dummy') +  #urgency
        te(18, 10,  # indication & ct
           lam=(2, 1.0),
           n_splines=(len(op['multi_cat_levels']['Indication']), 2),
           spline_order=(0, 0),
           dtype=('categorical', 'categorical')) +
        te(6, 11,  # pulse & ecg
           lam=(400, 2),
           n_splines=(20, 2),
           spline_order=(3, 0),
           dtype=('numerical', 'categorical')) 
    ).fit(*alb['pp']['train'].yield_train_X_y(i, trans_y=True)))

alb['gam'] = combine_mi_gams(alb_gams)


# Visualise partial dependence for each GAM feature:

# In[15]:


terms = alb['gam'].terms.info['terms'][:-1]

n_cis = 5
cis = np.linspace(0.025, 0.975, n_cis * 2)

n_rows = int(np.ceil((len(terms)) / 2))
fig, ax = plt.subplots(n_rows, 2, figsize=(8, 2.5 * n_rows))
ax = ax.ravel()

subplot_labels = {11: {'lines': ['No CT', 'CT'],
                       'title': 'Indication'},
                  12: {'lines': ['Normal ECG', 'Arrhythmia'],
                       'title': 'Heart rate'}}

for i, term in enumerate(terms):
    if term['term_type'] != 'tensor_term':
        XX = alb['gam'].generate_X_grid(term=i)
        pdep, confi = alb['gam'].partial_dependence(term=i, quantiles=cis)
        
        j = term['feature']
        
        for k in range(n_cis):
            ax[i].fill_between(XX[:, j],
                               confi[:, k][::-1],
                               confi[:, -(k + 1)][::-1],
                               alpha=1/n_cis, color='black', lw=0.0)
        
        ax[i].set_title(alb['pp']['train'].X['train'][0].columns[j])
    
    else:
        XX = alb['gam'].generate_X_grid(term=i, meshgrid=True)
        Z, confi = alb['gam'].partial_dependence(term=i, X=XX,
                                                 quantiles=cis,
                                                 meshgrid=True)
        
        if i in (11, 12):
            colours = ['blue', 'red']
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax[i].fill_between(XX[0][:, 0], confi[:, sli, k],
                                       confi[:, sli, -(k + 1)], lw=0.0,
                                       alpha=1/n_cis, color=colours[l])
                lines.append(Line2D([0], [0], color=colours[l]))

            ax[i].legend(lines, subplot_labels[i]['lines'],
                         loc='upper left')
            ax[i].set_title(subplot_labels[i]['title'])
        
plt.tight_layout()
plt.show()


# Make a more-informative plot for the indications feature:

# In[16]:


i = 11
colours = ['blue', 'red']
lines = []

j = alb['gam'].terms.info['terms'][i]['terms'][0]['feature']

XX = alb['gam'].generate_X_grid(
    term=i, n=len(op['multi_cat_levels']['Indication']), meshgrid=True)
Z, confi = alb['gam'].partial_dependence(term=i, X=XX,
                                         quantiles=cis,
                                         meshgrid=True)

fig, ax = plt.subplots(figsize=(9, 4))

for l, sli in enumerate([0, -1]):
    for k in range(n_cis):
        ax.fill_between(XX[0][:, 0], confi[:, sli, k],
                        confi[:, sli, -(k + 1)], lw=0.0,
                        alpha=1/n_cis, color=colours[l])
    lines.append(Line2D([0], [0], color=colours[l]))

ax.legend(lines, subplot_labels[i]['lines'], loc='upper left')
ax.set_title(subplot_labels[i]['title'])
ax.set_xticks(XX[0][:, j])
ax.set_xticklabels(op['multi_cat_levels']['Indication'],
                   rotation=45, rotation_mode='anchor',
                   horizontalalignment='right',
                   verticalalignment='top')
ax.set_xlim([XX[0][0, j], XX[0][-1, j]])
plt.grid(linewidth=0.5, linestyle=':')

plt.show()


# ### Sample new values and compare to test set

# In[17]:


compare_folds(alb)


# ## Impute lactate and albumin values for use in the train / test data that will be input to the mortality model

# In[18]:


for fold in ('train', 'test'):
    op[fold]['imp_all'] = copy.deepcopy(op[fold]['imp'])


# In[19]:


for d, name in ((lac, 'S03PreOpArterialBloodLactate'),
                (alb, 'S03PreOpLowestAlbumin')):
    for fold in ('train', 'test'):
        for i in pb(range(d['pp'][fold].n_imp_dfs),
                    prefix=f'{name} {fold}'):
            samples = quick_sample(
                d['gam'],
                d['pp'][fold].X['missing'][i],
                RANDOM_SEED, n_draws=1
            ).flatten()

            samples = d['trans'].inverse_transform(
                samples.reshape(-1, 1)).flatten()

            op[fold]['imp_all'][i].loc[
                op[fold]['imp_all'][i][name].isnull(), name] = samples


# In[20]:


for fold in ('train', 'test'):
    del op[fold]['imp_all'][51]


# In[21]:


for fold in ('train', 'test'):
    del op[fold]['imp']


# In[22]:


save_object(op, os.path.join('data', 'imputation_all_output.pkl'))