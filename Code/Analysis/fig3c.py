import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np


def year_product(data):
    bubble_scale = 5
    plt.figure(figsize=(6.8, 3), dpi=400)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 7})
    plt.xticks(rotation=45, fontsize=7, fontname='Arial', ha='right')
    plt.yticks(fontsize=7, fontname='Arial')
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    scatter = plt.scatter(
        data['year'],
        data['product'],
        s=data['num_papers'] * bubble_scale,
        alpha=0.7,
        c='#3690c0',
        edgecolors='black',
        linewidths=0.5
    )

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.7, num=5, color='#3690c0')
    sizes = np.linspace(min(data['num_papers']), max(data['num_papers']), num=5) * bubble_scale
    sizes[0] = 18.75
    custom_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='#3690c0',
            markeredgecolor='black',
            markersize=np.sqrt(size),
            alpha=0.7,
            linewidth=0.5
        )
        for size in sizes
    ]
    plt.legend(custom_handles, labels, loc="upper left", labelspacing=1, frameon=False)
    plt.xticks(range(2000, 2024))
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    plt.xlabel('Year')
    plt.savefig('../../Result/fig3_c.png', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    df_path = '../../Result/Database result/number_of_products_study_by_years.csv'
    df = pd.read_csv(df_path)
    plt.rcParams['font.family'] = 'Arial'
    year_product(pd.DataFrame(df))