#import sns

from utils.path_keeper import path_to_csv

# seaborn to check any corrilations between variables
scatter_fig = sns.pairplot(path_to_csv, kind="scatter", plot_kws={"alpha":0.4})