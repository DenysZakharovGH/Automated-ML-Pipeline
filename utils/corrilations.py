import sns

# seaborn to check any corrilations between variables
scatter_fig = sns.pairplot(csvFile, kind="scatter", plot_kws={"alpha":0.4})