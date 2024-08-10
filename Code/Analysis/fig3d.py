import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def heatmap(product_name_list, number_list):
    plt.figure(figsize=(8.5, 2), dpi=400)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)

    categories_y = ['E. coli', 'S. cerevisiae', 'C. glutamicum', 'Y. lipolytica', 'B. subtilis', 'P. pastoris',
                    'P. putida', 'C. acetobutylicum', 'K. marxianus', 'B. licheniformis']
    scores = np.array(number_list)

    row_sums = np.sum(scores, axis=1, keepdims=True)
    scores = scores / row_sums
    sns.heatmap(scores, cmap="YlGnBu", xticklabels=product_name_list, yticklabels=categories_y)
    ax = plt.gca()

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=1.5)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
    plt.xticks(rotation=45, fontsize=7, ha='right')
    plt.yticks(fontsize=7, fontstyle='italic')
    plt.xlabel('Target Product', fontsize=7)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    plt.savefig('../../Result/fig3_d.pdf', dpi=400, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    df_path = '../../Result/Database result/number_of_products_study_by_organisms.csv'
    df = pd.read_csv(df_path)
    strain_list=['E. coli', 'S. cerevisiae', 'C. glutamicum', 'Y. lipolytica', 'B. subtilis', 'P. pastoris',
     'P. putida', 'C. acetobutylicum', 'K. marxianus', 'B. licheniformis']
    product_name_list = df['product'][:30].tolist()
    number_list=[]
    for strain in strain_list:
        df_strain = df[df['organisms'] == strain]
        number_list.append(df_strain['number of study'].tolist())
    heatmap(product_name_list, number_list)