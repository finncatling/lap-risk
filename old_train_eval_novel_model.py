# ## Evaluate performance

# In[ ]:


# EVAL_FOLD = 'train'
EVAL_FOLD = "test"  # only use for final evaluation!!


# In[ ]:


imp_samples = []

for i in pb(range(len(op[EVAL_FOLD]["imp_all"]))):
    imp_samples.append(
        quick_sample(
            gam,
            op[EVAL_FOLD]["imp_all"][i].values,
            RANDOM_SEED,
            quantity="mu",
            n_draws=50,
        )
    )

samples = np.vstack(imp_samples)
samples.shape


# Inspect some risk distributions for individual patients:

# In[ ]:


for i in range(10, 20):
    plt.hist(samples[:, i], alpha=0.3)
plt.xlim((-0.03, 1.03))
plt.show()


# Look at overall distribution of predicted risks:

# In[ ]:


hist_args = {"bins": 50, "density": True, "alpha": 0.5}

for i, outcome in enumerate(("lived", "died")):
    strat_samples = samples[:, np.where(op[EVAL_FOLD]["y"] == i)[0]].flatten()
    plt.hist(strat_samples, label=outcome, **hist_args)

plt.legend()
plt.show()


# In[ ]:


y_point_pred = np.median(samples, 0)


# In[ ]:


evaluate_predictions(op[EVAL_FOLD]["y"], y_point_pred)


# ## Figures for ASA abstract

# ### Illustrating distributions of predicted risks vs. point predictions

# In[ ]:


from arviz.plots import plot_kde


# In[ ]:


p95 = np.percentile(samples, (2.5, 97.5), 0).T
p95_range = p95[:, 1] - p95[:, 0]


# In[ ]:


n = 5 * 4
fig, ax = plt.subplots(5, 4, figsize=(10, 10))
ax = ax.ravel()

for i, j in enumerate(np.argpartition(p95_range, -n)[-n:]):
    ax[i].set(xlim=(0, 1), title=j)
    ax[i].hist(samples[:, j], bins=30)

plt.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9, 5))
plt.rcParams.update({"font.size": 16})

hist_args = {"bins": 50, "range": (0, 1), "density": True, "alpha": 1.0}
axvline_args = {"color": "black", "ls": "--"}

for i, j in enumerate((13672, 1)):
    ax.hist(samples[:, j], label=f"Patient {i + 1}", **hist_args)
    if i:
        ax.axvline(np.median(samples[:, j]), label="Point prediction", **axvline_args)
    else:
        ax.axvline(np.median(samples[:, j]), **axvline_args)


ax.set(xlim=(0, 1), xlabel="Predicted mortality risk", ylabel="Probability density")
ax.legend()
plt.show()
