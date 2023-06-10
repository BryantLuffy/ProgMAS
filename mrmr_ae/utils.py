def do_km_plot(survive_data, pvalue, cindex, cancer_type, model_name):
    # import necessary packages
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # extract relevant data
    values = np.asarray(survive_data['Type'])
    events = np.asarray(survive_data['OS'])
    times = np.asarray(survive_data['OS.time'])

    # set plotting style
    sns.set(style='ticks', context='notebook', font_scale=1.5)

    # create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # customize plot style
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)  # set thickness of x-axis line
    ax.spines['left'].set_linewidth(1.5)  # set thickness of y-axis line

    # fit and plot Kaplan-Meier survival curves for each cluster
    kaplan = KaplanMeierFitter()
    for label in set(values):
        kaplan.fit(times[values == label],
                   event_observed=events[values == label],
                   label='cluster {0}'.format(label))
        kaplan.plot_survival_function(ax=ax, ci_alpha=0)
        ax.legend(loc=1, frameon=False)

    # customize plot labels and title based on whether C-index was calculated or not
    if cindex == None:
        ax.set_xlabel('days', fontsize=20)
        ax.set_ylabel('Survival Probability', fontsize=20)
        ax.set_title('{1} \n Cancer: {0}    p-value.:{2: .1e} '.format(
            cancer_type, model_name, pvalue),
            fontsize=18,
            fontweight='bold')
    else:
        ax.set_xlabel('days', fontsize=20)
        ax.set_title('{1} \n Cancer: {0}  p-value.:{2: .1e}   Cindex: {3: .2f}'.format(
            cancer_type, model_name, pvalue, cindex),
            fontsize=18,
            fontweight='bold')

    # save plot as a .tiff file
    fig.savefig('./' + str(cancer_type) + model_name + '.tiff', dpi=300)




def add_survival_labels(data):
    data['label'] = 0
    data.loc[(data['OS.time'] > 1825) & (data['OS'].isin([0, 1])), 'label'] = 1
    data = data.loc[~((data['OS.time'] < 1825) & (data['OS'] == 0))]
    return data