import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def heatmap(x_list,y_list,data):
    plt.figure(figsize=(8.75, 3), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.yticks(fontsize=7)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    y_list = [name.capitalize() for name in y_list]
    categories_x = x_list
    categories_y = y_list
    scores = np.array(data)
    sns.heatmap(scores, cmap="YlGnBu", xticklabels=categories_x, yticklabels=categories_y)
    ax = plt.gca()
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=1.5)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1)
    plt.tick_params(axis='x', direction='in', width=0.5, which='both', length=1)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel('Target Product', fontsize=7)
    plt.savefig('../../Result/figS2_b.pdf', dpi=400, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    df_path = '../../Result/Database result/number_of_gene_modifications_ecoli.csv'
    df = pd.read_csv(df_path)
    product_name = ['acetate', 'ethanol', 'succinic acid', 'lactic acid', 'lycopene', 'acetic acid', 'Pyruvic acid', '2,3-butanediol', 'L-phenylalanine', 'isobutanol', '1-butanol', 'succinate', 'Tryptophan', 'L-tyrosine', 'shikimate', 'p-coumaric acid', '5-Aminolevulinic acid', 'formate', '3-hydroxypropionic acid', 'acetoin']
    knock_list = []
    knock_result = []
    for product in product_name:
        df_product = df[df['product'] == product]
        knock_result.append(df_product['number of article'].tolist())
        knock_list = df_product['gene'].tolist()
    heatmap(x_list=knock_list,y_list=product_name,data=knock_result)