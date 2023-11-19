import seaborn as sns
import matplotlib.pyplot as plt


def his(x):
    for i in range(7):
        plt.figure(figsize=(6, 4))
        sns.histplot(x[:, i])
        plt.savefig(f'histogram/his_c_{i}.pdf')








