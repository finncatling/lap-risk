#!/usr/bin/env python
# coding: utf-8

# In[136]:


import os, sys
import copy
sys.path.append(os.pardir)
import numpy as np
from pprint import PrettyPrinter
import matplotlib.pyplot as plt

from utils.io import load_object
from utils.model.novel import NovelModel
from utils.constants import NOVEL_MODEL_OUTPUT_DIR
from utils.gam import quick_sample


# In[7]:


novel_model: NovelModel = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        '13_novel_model_production.pkl'
    )
)


# ## Initialise dict for novel model assets export

# In[75]:


assets = {
    'albumin': {'model': copy.deepcopy(novel_model.alb_imputer.imputers[0])},
    'lactate': {'model': copy.deepcopy(novel_model.lac_imputer.imputers[0])},
    'mortality': {'model': novel_model.models[0]}
}


# ## Winsor thresholds

# In[76]:


wt = copy.deepcopy(novel_model.cat_imputer.swm.winsor_thresholds[0])


# In[77]:


wt[novel_model.alb_imputer.lacalb_variable_name] = copy.deepcopy(
    copy.deepcopy(novel_model.alb_imputer.winsor_thresholds[0])
)

wt[novel_model.lac_imputer.lacalb_variable_name] = copy.deepcopy(
    copy.deepcopy(novel_model.lac_imputer.winsor_thresholds[0])
)


# In[78]:


wt['S01AgeOnArrival'][0] = 18.0


# In[79]:


del wt['S03GlasgowComaScore']


# In[80]:


assets['winsor_thresholds'] = wt


# ## Albumin imputer assets

# In[99]:


alb_features = novel_model.alb_imputer._get_features_where_lacalb_missing(
    fold_name='train',
    split_i=0,
    mice_imp_i=0
)


# In[100]:


assets['albumin']['input_data'] = {
    'dtypes': copy.deepcopy(alb_features.dtypes),
    'describe': alb_features.describe(),
    'unique_categories': {}
}


# In[101]:


for c in alb_features[alb_features.columns.difference(list(wt.keys()))].columns:
    assets['albumin']['input_data']['unique_categories'][c] = np.sort(
        alb_features[c].unique()
    )


# In[102]:


assets['albumin']['transformer'] = novel_model.alb_imputer.transformers[0]


# ## Lactate imputer assets

# In[106]:


lac_features = novel_model.lac_imputer._get_features_where_lacalb_missing(
    fold_name='train',
    split_i=0,
    mice_imp_i=0
)


# In[107]:


assets['lactate']['input_data'] = {
    'dtypes': copy.deepcopy(lac_features.dtypes),
    'describe': lac_features.describe(),
    'unique_categories': {}
}


# In[108]:


for c in lac_features[lac_features.columns.difference(list(wt.keys()))].columns:
    assets['lactate']['input_data']['unique_categories'][c] = np.sort(
        lac_features[c].unique()
    )


# In[109]:


assets['lactate']['transformer'] = novel_model.lac_imputer.transformers[0]


# ## Mortality model assets

# In[110]:


mort_features, mort_labels = novel_model.get_features_and_labels(
    fold_name='train',
    split_i=0,
    mice_imp_i=0,
    lac_alb_imp_i=0
)


# In[115]:


assets['mortality']['input_data'] = {
    'dtypes': copy.deepcopy(mort_features.dtypes),
    'describe': mort_features.describe(),
    'unique_categories': {}
}


# In[116]:


for c in mort_features[mort_features.columns.difference(list(wt.keys()))].columns:
    assets['mortality']['input_data']['unique_categories'][c] = np.sort(
        mort_features[c].unique()
    )


# ## Label encoding for non-binary categorical variables

# In[122]:


assets['label_encoding'] = load_object(
    os.path.join(
        NOVEL_MODEL_OUTPUT_DIR,
        "05_multi_category_levels_with_indications.pkl"
    )
)


# ## Sanity check

# ### No imputation

# In[132]:


mort_pred = quick_sample(
    gam = assets['mortality']['model'],
    sample_at_X = mort_features.iloc[:1, :].values,
    quantity='mu',
    n_draws=500,
    random_seed=1
)


# In[138]:


plt.hist(mort_pred.flatten())
plt.show()


# ### Imputation

# In[ ]:





# ## Inspect assets

# In[123]:


PrettyPrinter().pprint(assets)

