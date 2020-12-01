#!/usr/bin/env python
# coding: utf-8

# # Plots to display novel model output

# In[42]:


import os, sys
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arviz.stats.density_utils import kde, _kde_linear

from utils.io import load_object
from utils.model.novel import NovelModel
from utils.constants import NOVEL_MODEL_OUTPUT_DIR
from utils.evaluate import stratify_y_pred
from utils.gam import quick_sample

pd.set_option('display.max_columns', 50)


# In[2]:


novel_model: NovelModel = load_object(
    os.path.join(NOVEL_MODEL_OUTPUT_DIR, "08_novel_model.pkl"))


# In[3]:


y_true, y_pred = novel_model.get_observed_and_predicted('test', 0, 50)


# ## Inspect risk distributions for individual patients

# In[4]:


fig, ax = plt.subplots()

for i in range(15, 20):
    ax.hist(
        y_pred[:, i],
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5)

ax.set(
    xlim=(0, 1),
    xlabel='Risk of death',
    ylabel='Probability density'
)

plt.show()


# In[5]:


fig, ax = plt.subplots()

for i in range(15, 20):
    grid, pdf = kde(
        y_pred[:, i],
        bw='silverman',
#         custom_lims=(0, 1)
    )
    ax.plot(grid, pdf)

ax.set_ylim(bottom=0)
ax.set(
    xlim=(0, 1),
    xlabel='Risk of death',
    ylabel='Probability density'
)

plt.show()


# For those patients with very low predicted risks, the histogram doesn't look Gaussian (more like half Gaussian), so forcing a Gaussian KDE to be fit seems to be distorting the y scale. Probably best to use histograms.

# Inspect widths of mortality risk distributions across all patients:

# In[6]:


fig, ax = plt.subplots()

for i, label in enumerate(('lived', 'died')):
    stratum = y_pred[:, np.where(y_true == i)[0]]
    p95 = np.percentile(stratum, (2.5, 97.5), 0).T
    p95_range = p95[:, 1] - p95[:, 0]
    ax.hist(
        p95_range,
        bins=30,
        label=label,
        alpha=0.5,
        density=True,
        range=(0, 0.6)
    )

ax.set(xlabel='95% range')
ax.set_xlim(left=0)
ax.legend()
plt.show()


# In[28]:


subplot_grid_dims = (4, 4)
n_patients = np.product(subplot_grid_dims)
chosen = np.random.choice(y_true.size, n_patients, replace=False)

fig, ax = plt.subplots(
    *subplot_grid_dims,
    figsize=(subplot_grid_dims[0] * 2, subplot_grid_dims[1] * 1.6)
)
ax = ax.ravel()

for i, j in enumerate(chosen):
    ax[i].set(title=j, xlim=(0, 1))
    ax[i].hist(y_pred[:, j], bins=50, range=(0, 1), density=True)

plt.tight_layout()
plt.show()


# 9429

# In[36]:


fig, ax = plt.subplots()

# for i in (9429, 1580, 337, 14817):
# for i in (9942, 337):
# for i in (9942, 1580):
# for i in (9429, 20116):
for i in (9942, 3094):
    ax.hist(
        y_pred[:, i],
        bins=80,
        range=(0, 1),
        density=True,
        alpha=0.5)

ax.set(
    xlim=(0, 1),
    xlabel='Risk of death',
    ylabel='Probability density'
)

plt.show()


# In[86]:


fig, ax = plt.subplots()

# bws = (0.008, 0.04)
bws = (0.008, 0.012, 0.04)

for i, j in enumerate((9942, 6530, 3094)):
    grid, pdf = kde(
        y_pred[:, j],
        bw=bws[i],
#         adaptive=True
#         custom_lims=(0, 1)
    )
#     ax.plot(grid, pdf)
    ax.fill_between(grid, pdf, alpha=0.4, label=f'Patient {i + 1}')
    if not i:
        ax.axvline(
            np.median(y_pred[:, j]), c='black', ls=':', label='Point prediction')
    else:
        ax.axvline(np.median(y_pred[:, j]), c='black', ls=':')

ax.set_ylim(bottom=0)
ax.set(
    xlim=(0, 1),
    xlabel='Predicted risk of death',
    ylabel='Probability density'
)
ax.legend()

plt.show()


# Restrict just to patients who died:

# In[29]:


np.where((y_true == 1).astype(int).cumsum() == 904)[0][0]


# In[21]:


subplot_grid_dims = (12, 8)
n_patients = np.product(subplot_grid_dims)
died = y_pred[:, np.where(y_true == 1)[0]]
chosen = np.random.choice(died.shape[1], n_patients, replace=False)

fig, ax = plt.subplots(
    *subplot_grid_dims,
    figsize=(subplot_grid_dims[1] * 2, subplot_grid_dims[0] * 1.5)
)
ax = ax.ravel()

y_true_cumsum = (y_true == 1).astype(int).cumsum()

for i, j in enumerate(chosen):
    title = np.where(y_true_cumsum == j + 1)[0][0]
    ax[i].set(title=title, xlim=(0, 1))
    ax[i].hist(died[:, j], bins=50, range=(0, 1), density=True)

plt.tight_layout()
plt.show()


# Good example patients: 397, 991, 1752, 656, 1394, 133, 2232, 499, 903.
# 
# 903 is bimodal and may be best as an example where both:
# 
# - the median is a poor summary statistic
# - distribution spans a wide range of predicted mortalities

# In[74]:


fig, ax = plt.subplots(figsize=(9, 5))

hist_args = {'bins': 50,
             'range': (0, 1),
             'density': True,
             'alpha': 1.0}
axvline_args = {'color': 'black', 'ls': '--'}

for i, j in enumerate((13672, 1)):
    ax.hist(samples[:, j], label=f'Patient {i + 1}', **hist_args)
    if i:
        ax.axvline(np.median(samples[:, j]), label='Point prediction',
                   **axvline_args)
    else:
        ax.axvline(np.median(samples[:, j]), **axvline_args)
        

ax.set(xlim=(0, 1),
       xlabel='Predicted mortality risk',
       ylabel='Probability density')
ax.legend()
plt.show()


# ## Figures for ASA presentation

# In[ ]:


import arviz as az


# ### Find a good patient to use as an example

# In[ ]:


n = 5 * 4
fig, ax = plt.subplots(5, 4, figsize=(10, 10))
ax = ax.ravel()

for i, j in enumerate(np.argpartition(p95_range, -n)[-120:-100]):
    ax[i].set(xlim=(0, 1), title=j)
    ax[i].hist(samples[:, j], bins=50)

plt.tight_layout()
plt.show()


# In[ ]:


example_pt_i = 578
# example_pt_i = 10061
# example_pt_i = 9212
# example_pt_i = 13672


# In[ ]:


kde = az.plot_kde(samples[:, example_pt_i], bw=7)


# In[ ]:


def plot_saver(
    fig,
    output_filename: str,
    output_dir: str = os.path.join(os.pardir, 'lap-risk-outputs',
                                   'figures', 'asa2020'),
    extensions: Tuple[str] = ("pdf", "png")
) -> None:
    for ext in extensions:
        fig.savefig(
            os.path.join(output_dir, f"{output_filename}.{ext}"),
            format=ext,
            bbox_inches="tight",
            dpi=300)


# In[ ]:


example_pt_median_risk = np.median(samples[:, example_pt_i])
print(f'Median mortality risk = {example_pt_median_risk}')

for i, plot_label in enumerate(['mortality_point_estimate',
                                'mortality_point_estimate_and_distribution']):

    fig, ax = plt.subplots(figsize=(10, 5 * 10/9))
    plt.rcParams.update({'font.size': 18})

    axvline_args = {'color': 'black', 'ls': ':', 'linewidth': 3}
    
    if i:
        ax.fill_between(kde.lines[0]._x,
                        kde.lines[0]._y,
                        np.zeros_like(kde.lines[0]._y),
                        alpha=0.5)
    
    ax.axvline(example_pt_median_risk,
               label='Point prediction',
               **axvline_args)

    ax.set(xlim=(0, 1),
           ylim=(0, 4),
           yticks=np.arange(5),
           xlabel='Predicted mortality risk',
           ylabel='Probability density')

    plot_saver(fig, f'{plot_label}_{example_pt_i}')
    plt.show()


# ## Compare uncertainty in predictions where lactate / albumin missing or present

# In[ ]:


op['test'].keys()


# ### Inspect non-missing lactates and albumins in the train data

# In[ ]:


lactates = op['train']['X_df'].loc[
    op['train']['X_df']['S03PreOpArterialBloodLactate'].notnull(),
    'S03PreOpArterialBloodLactate'].values

albumins = op['train']['X_df'].loc[
    op['train']['X_df']['S03PreOpLowestAlbumin'].notnull(),
    'S03PreOpLowestAlbumin'].values


# In[ ]:


plt.hist(lactates, bins=50)
plt.show()


# In[ ]:


plt.hist(albumins, bins=20)
plt.show()


# Grab the median values for comparison with the imputed values (and consequent risk distributions) later:

# In[ ]:


median_train_lactate = np.median(lactates)
median_train_albumin = np.median(albumins)
print(median_train_lactate, median_train_albumin)


# ### Recover the imputed albumins / lactates for test-set patients where both are missing

# In[ ]:


la_test_missing_i = op['test']['X_df'].loc[
    (op['test']['X_df']['S03PreOpArterialBloodLactate_missing'] == 1.0) &
    (op['test']['X_df']['S03PreOpLowestAlbumin_missing'] == 1.0)
].index


# In[ ]:


example_pt_i in la_test_missing_i


# In[ ]:


# 3D array to hold imputed values. In last dimension lactate is in 0, albumin in 1
imp_test_lac_alb = np.zeros([len(op['test']['imp_all']),
                             la_test_missing_i.shape[0],
                             2])


# In[ ]:


for i in range(len(op['test']['imp_all'])):
    imp_test_lac_alb[i, :, :] = op['test']['imp_all'][i].loc[
        la_test_missing_i,
        ['S03PreOpArterialBloodLactate', 'S03PreOpLowestAlbumin']].values


# In[ ]:


la_test_missing_i.get_loc(example_pt_i)


# In[ ]:


op['test']['X_df'].loc[example_pt_i]


# Note that the predicted lactates look high for a patient with these numbers! But it turns out they have CT-confirmed ischaemic bowel...

# In[ ]:


op['multi_cat_levels']['Indication'][5]


# In[ ]:


for i, label in enumerate(['lactate', 'albumin']):
    plt.hist(imp_test_lac_alb[:, la_test_missing_i.get_loc(example_pt_i), i])
    plt.xlabel(label)
    plt.show()


# In[ ]:


lactate_kde = az.plot_kde(imp_test_lac_alb[:, la_test_missing_i.get_loc(example_pt_i), 0],
                          bw=9)


# In[ ]:


albumin_kde = az.plot_kde(imp_test_lac_alb[:, la_test_missing_i.get_loc(example_pt_i), 1],
                          bw=8)


# In[ ]:


lactate = {'kde': lactate_kde,
           'train_median': median_train_lactate,
           'label': 'Lactate (mmol/L)',
           'xlim': (0, 20),
           'ylim': (0, 0.21)}

albumin = {'kde': albumin_kde,
           'train_median': median_train_albumin,
           'label': 'Albumin (g/L)',
           'xlim': (10, 52),
           'ylim': (0, 0.055)}


# In[ ]:


for x in (lactate, albumin):
    for i, plot_label in enumerate(['median', 'median_and_patient_dist']): 
        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': 18})
        
        if i:
            ax.fill_between(x['kde'].lines[0]._x,
                            x['kde'].lines[0]._y,
                            np.zeros_like(kde.lines[0]._y),
                            alpha=0.5)
        ax.axvline(x['train_median'], **axvline_args)

        ax.set(xlabel=x['label'], ylabel='Probability density',
               xlim=x['xlim'], ylim=x['ylim'])
        
        plot_saver(fig, f'{x["label"].split()[0].lower()}_{plot_label}_{example_pt_i}')
        plt.show()


# Repredict the mortality risk if impute the population medians for lactate and albumin:

# In[ ]:


example_pt_samples = []

for i in pb(range(len(op['test']['imp_all']))):
    example_pt_imp = op['test']['imp_all'][i].loc[example_pt_i].copy()
    
    example_pt_imp['S03PreOpArterialBloodLactate'] = median_train_lactate
    example_pt_imp['S03PreOpLowestAlbumin'] = median_train_albumin
    
    example_pt_samples.append(quick_sample(gam,
                                           example_pt_imp.values.reshape(1, 23),
                                           RANDOM_SEED,
                                           quantity='mu',
                                           n_draws=50))

example_patient_samples = np.vstack(example_pt_samples)
example_patient_samples.shape


# NB. perhaps should change missingness indicators above?

# In[ ]:


median_lacalb_kde = az.plot_kde(
    example_patient_samples[:, 0],
    bw=30)


# In[ ]:


fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 18})

ax.fill_between(median_lacalb_kde.lines[0]._x,
                median_lacalb_kde.lines[0]._y,
                np.zeros_like(kde.lines[0]._y),
                alpha=0.5)

# ax.fill_between(kde.lines[0]._x,
#                 kde.lines[0]._y,
#                 np.zeros_like(kde.lines[0]._y),
#                 alpha=0.5)

ax.set(xlim=(0, 1),
       ylim=(0, 13.5),
       xlabel='Predicted mortality risk',
       ylabel='Probability density')

plot_saver(fig, f'example_pt_mortality_median_lac_alb_{example_pt_i}')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 18})

ax.fill_between(median_lacalb_kde.lines[0]._x,
                median_lacalb_kde.lines[0]._y,
                np.zeros_like(kde.lines[0]._y),
                color='lightgray',
                alpha=0.5)

ax.fill_between(kde.lines[0]._x,
                kde.lines[0]._y,
                np.zeros_like(kde.lines[0]._y),
                alpha=0.5)

ax.set(xlim=(0, 1),
       ylim=(0, 13.5),
       xlabel='Predicted mortality risk',
       ylabel='Probability density')

plot_saver(fig, f'example_pt_mortality_median_lac_alb_comparison_{example_pt_i}')
plt.show()


# ### Partial dependence plots

# In[ ]:


aes_i = [0, 8, 2, 6]
aes_subset = [aes[i] for i in aes_i]

aes_subset[0].gs_pos = (0, 0)
aes_subset[1].gs_pos = (0, 1)
aes_subset[2].gs_pos = (1, 0)
aes_subset[3].gs_pos = (1, 1)

terms = gam.terms.info['terms'][:-1]
terms_subset = [terms[i] for i in aes_i]

n_cols = 2
row_height = 3.5
ticks_per_cat = 21
scaler = 1.0
plt.rcParams.update({'font.size': 18})

mid_cat_i = int((ticks_per_cat - 1) / 2)

n_rows = int(np.ceil((len(terms_subset)) / n_cols))
fig = plt.figure(figsize=(scaler * 12, scaler * row_height * n_rows))
gs = fig.add_gridspec(n_rows, n_cols)

for i, term in enumerate(terms_subset):
    if aes_subset[i].view_3d is None:
        ax = fig.add_subplot(gs[aes_subset[i].gs_pos])
        ax.set_title(aes_subset[i].pretty_name)
    else:
        ax = fig.add_subplot(gs[aes_subset[i].gs_pos], projection='3d')
    
    if aes_subset[i].labels is None:
        n = 100
    else:
        n = len(aes_subset[i].labels) * ticks_per_cat
    
    if term['term_type'] != 'tensor_term':        
        XX = gam.generate_X_grid(term=aes_i[i], n=n)
        pdep, confi = gam.partial_dependence(term=aes_i[i], X=XX,
                                             quantiles=cis)
        j = term['feature']
        
        for k in range(n_cis):
            ax.fill_between(XX[:, j],
                            confi[:, k], confi[:, -(k + 1)],
                            alpha=1/n_cis, color='tab:blue', lw=0.0)
        
        if aes_subset[i].labels is not None:
            ax.set_xticks(XX[:, j][range(mid_cat_i, n, ticks_per_cat)])
            ax.set_xticklabels(aes_subset[i].labels)
    
    else:
        print('hi')
        XX = gam.generate_X_grid(term=aes_i[i], n=n, meshgrid=True)        
        Z, confi = gam.partial_dependence(term=aes_i[i], X=XX,
                                          quantiles=cis,
                                          meshgrid=True)       

        if aes_subset[i].view_3d is None:
            colours = ['tab:blue', 'tab:orange']
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax.fill_between(XX[0][:, 0], confi[:, sli, k],
                                    confi[:, sli, -(k + 1)], lw=0.0,
                                    alpha=1/n_cis, color=colours[l])
                    
                lines.append(Line2D([0], [0], color=colours[l]))
            ax.legend(lines, aes_subset[i].strata, loc=aes_subset[i].legend_loc)
                    
            if aes_subset[i].labels is not None:
                ax.set_xticks(XX[0][:, 0][range(mid_cat_i, n, ticks_per_cat)])
                ax.set_xticklabels(aes_subset[i].labels)
                ax.set_xlim([XX[0][0, 0], XX[0][-1, 0]])
                if aes_subset[i].name == 'Indication':
                    ax.set_xticklabels(aes_subset[i].labels,
                                       rotation=45, rotation_mode='anchor',
                                       horizontalalignment='right',
                                       verticalalignment='top')
            
        else:
            ax.plot_surface(XX[0], XX[1], Z, cmap='Blues')
            ax.view_init(*aes_subset[i].view_3d)
            ax.set_xlabel(aes_subset[i].pretty_name[0])
            ax.set_ylabel(aes_subset[i].pretty_name[1])

fig.text(0, 0.5, 'logit( probability of death )',
         rotation='vertical', va='center')

plt.tight_layout()
plot_saver(fig, 'age_asa_k_lactate')
plt.show()


# In[ ]:


aes_i = [11, 12, 13]
aes_subset = [aes[i] for i in aes_i]

aes_subset[0].gs_pos = (0, 0)
aes_subset[1].gs_pos = (0, 1)
aes_subset[2].gs_pos = (slice(1, 3), slice(0, 2))

aes_subset[1].labels = ['None', 'Serous', 'Local\npus',
                        'Free pus /\nblood / faeces']

short_inds = [sanitize_ind(s) for s in
              op['multi_cat_levels']['Indication']]
short_inds[13] = 'SBO + hernia'
short_inds[14] = 'Perforation + abscess'
short_inds[15] = 'SBO + ischaemia'
short_inds[16] = 'Obstruction + ischaemia'
short_inds[17] = 'SBO + LBO'
short_inds[18] = 'Peritonitis + leak'
short_inds[19] = 'Peritonitis + ischaemia'
short_inds[20] = 'Peritonitis + abscess'
aes_subset[2].labels = short_inds

terms = gam.terms.info['terms'][:-1]
terms_subset = [terms[i] for i in aes_i]

n_cols = 2
row_height = 4
ticks_per_cat = 21
scaler = 1.2
plt.rcParams.update({'font.size': 18})

mid_cat_i = int((ticks_per_cat - 1) / 2)

n_rows = int(np.ceil((len(terms_subset)) / n_cols))
fig = plt.figure(figsize=(scaler * 12, scaler * row_height * n_rows))
gs = fig.add_gridspec(n_rows, n_cols)

for i, term in enumerate(terms_subset):
    if aes_subset[i].view_3d is None:
        ax = fig.add_subplot(gs[aes_subset[i].gs_pos])
        ax.set_title(aes_subset[i].pretty_name)
    else:
        ax = fig.add_subplot(gs[aes_subset[i].gs_pos], projection='3d')
    
    if aes_subset[i].labels is None:
        n = 100
    else:
        n = len(aes_subset[i].labels) * ticks_per_cat
    
    if term['term_type'] != 'tensor_term':        
        XX = gam.generate_X_grid(term=aes_i[i], n=n)
        pdep, confi = gam.partial_dependence(term=aes_i[i], X=XX,
                                             quantiles=cis)
        j = term['feature']
        
        for k in range(n_cis):
            ax.fill_between(XX[:, j],
                            confi[:, k], confi[:, -(k + 1)],
                            alpha=1/n_cis, color='tab:blue', lw=0.0)
        
        if aes_subset[i].labels is not None:
            ax.set_xticks(XX[:, j][range(mid_cat_i, n, ticks_per_cat)])
            ax.set_xticklabels(aes_subset[i].labels)
    
    else:
        XX = gam.generate_X_grid(term=aes_i[i], n=n, meshgrid=True)        
        Z, confi = gam.partial_dependence(term=aes_i[i], X=XX,
                                          quantiles=cis,
                                          meshgrid=True)       

        if aes_subset[i].view_3d is None:
            colours = ['tab:blue', 'tab:orange']
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax.fill_between(XX[0][:, 0], confi[:, sli, k],
                                    confi[:, sli, -(k + 1)], lw=0.0,
                                    alpha=1/n_cis, color=colours[l])
                    
                lines.append(Line2D([0], [0], color=colours[l]))
            ax.legend(lines, aes_subset[i].strata, loc=aes_subset[i].legend_loc)
                    
            if aes_subset[i].labels is not None:
                ax.set_xticks(XX[0][:, 0][range(mid_cat_i, n, ticks_per_cat)])
                ax.set_xticklabels(aes_subset[i].labels)
                ax.set_xlim([XX[0][0, 0], XX[0][-1, 0]])
                if aes_subset[i].name == 'Indication':
                    ax.set_xticklabels(
                        aes_subset[i].labels,
                        rotation=33, rotation_mode='anchor',
                        horizontalalignment='right',
                        verticalalignment='top')
            
        else:
            ax.plot_surface(XX[0], XX[1], Z, cmap='Blues')
            ax.view_init(*aes_subset[i].view_3d)
            ax.set_xlabel(aes_subset[i].pretty_name[0])
            ax.set_ylabel(aes_subset[i].pretty_name[1])

fig.text(0.09, 0.59, 'logit( probability of death )',
         rotation='vertical', va='center')

plt.tight_layout()
plot_saver(fig, 'malig_peritsoil_indications')
plt.show()


# In[ ]:


aes_i = [3, 9, 16]
aes_subset = [aes[i] for i in aes_i]

aes_subset[0].gs_pos = (0, 0)
aes_subset[1].gs_pos = (0, 1)
aes_subset[2].gs_pos = (0, 2)

terms = gam.terms.info['terms'][:-1]
terms_subset = [terms[i] for i in aes_i]

n_cols = 3
row_height = 3.3
ticks_per_cat = 21
scaler = 1.2
plt.rcParams.update({'font.size': 16})

mid_cat_i = int((ticks_per_cat - 1) / 2)

n_rows = int(np.ceil((len(terms_subset)) / n_cols))
fig = plt.figure(figsize=(scaler * 12, scaler * row_height * n_rows))
gs = fig.add_gridspec(n_rows, n_cols)

for i, term in enumerate(terms_subset):
    if aes_subset[i].view_3d is None:
        ax = fig.add_subplot(gs[aes_subset[i].gs_pos])
        ax.set_title(aes_subset[i].pretty_name)
    else:
        ax = fig.add_subplot(gs[aes_subset[i].gs_pos], projection='3d')
    
    if aes_subset[i].labels is None:
        n = 100
    else:
        n = len(aes_subset[i].labels) * ticks_per_cat
    
    if term['term_type'] != 'tensor_term':        
        XX = gam.generate_X_grid(term=aes_i[i], n=n)
        pdep, confi = gam.partial_dependence(term=aes_i[i], X=XX,
                                             quantiles=cis)
        j = term['feature']
        
        for k in range(n_cis):
            ax.fill_between(XX[:, j],
                            confi[:, k], confi[:, -(k + 1)],
                            alpha=1/n_cis, color='tab:blue', lw=0.0)
        
        if aes_subset[i].labels is not None:
            ax.set_xticks(XX[:, j][range(mid_cat_i, n, ticks_per_cat)])
            ax.set_xticklabels(aes_subset[i].labels)
    
    else:
        XX = gam.generate_X_grid(term=aes_i[i], n=n, meshgrid=True)        
        Z, confi = gam.partial_dependence(term=aes_i[i], X=XX,
                                          quantiles=cis,
                                          meshgrid=True)       

        if aes_subset[i].view_3d is None:
            colours = ['tab:blue', 'tab:orange']
            lines = []

            for l, sli in enumerate([0, -1]):
                for k in range(n_cis):
                    ax.fill_between(XX[0][:, 0], confi[:, sli, k],
                                    confi[:, sli, -(k + 1)], lw=0.0,
                                    alpha=1/n_cis, color=colours[l])
                    
                lines.append(Line2D([0], [0], color=colours[l]))
            ax.legend(lines, aes_subset[i].strata, loc=aes_subset[i].legend_loc)
                    
            if aes_subset[i].labels is not None:
                ax.set_xticks(XX[0][:, 0][range(mid_cat_i, n, ticks_per_cat)])
                ax.set_xticklabels(aes_subset[i].labels)
                ax.set_xlim([XX[0][0, 0], XX[0][-1, 0]])
                if aes_subset[i].name == 'Indication':
                    ax.set_xticklabels(
                        aes_subset[i].labels,
                        rotation=33, rotation_mode='anchor',
                        horizontalalignment='right',
                        verticalalignment='top')
            
        else:
            ax.plot_surface(XX[0], XX[1], Z, cmap='Blues')
            ax.view_init(*aes_subset[i].view_3d)
            ax.set_xlabel(aes_subset[i].pretty_name[0])
            ax.set_ylabel(aes_subset[i].pretty_name[1])

fig.text(0, 0.5, 'logit( probability of death )',
         rotation='vertical', va='center')

plt.tight_layout()

subplot3d_pos = ax.get_position()
subplot3d_pos = [
    subplot3d_pos.x0 - 0.05,
    subplot3d_pos.y0 + 0.08,
    subplot3d_pos.width,
    subplot3d_pos.height
]
ax.set_position(subplot3d_pos)

plot_saver(fig, 'wcc_lactatemissing_ureacreat')
plt.show()

