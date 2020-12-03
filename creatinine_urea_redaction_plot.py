f, axs = plt.subplots(1, 2, figsize=(8.1, 4))
axs = axs.ravel()
titles = ['Raw data', 'After redaction']

for i, data in enumerate([df, redact]):
    axs[i].scatter(
        data['S03SerumCreatinine'].values,
        data['S03Urea'].values,
        alpha=0.1,
        s=5
    )
    axs[i].set(
        xlabel='Creatinine (mmol/L)',
        ylabel='Urea (mmol/L)',
        title=titles[i],
        xlim=(-35, 1235),
        ylim=(-10, 310)
    )

plt.tight_layout()
plt.show()

