from compare_laser import compare_d2cell, compare_other
import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


def heatmap(x_list,y_list,data):
    plt.figure(figsize=(3.75, 1.5), dpi=400)
    # plt.figure(figsize=(7, 4), dpi=400)
    plt.rcParams.update({'font.size': 8})
    plt.yticks(fontsize=8)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    # y_list = [name.capitalize() for name in y_list]
    categories_x = x_list
    categories_y = y_list

    scores = np.array(data)

    # create heatmap
    sns.heatmap(scores, cmap="Blues", annot=True, fmt=".1f", linewidths=.5,
                vmin=60, vmax=95, xticklabels=categories_x, yticklabels=categories_y, annot_kws={'size': 7},
                linecolor='black')
    ax = plt.gca()
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=1.5)
    cbar.ax.set_ylabel('Recall (%)', fontsize=8, rotation=270, labelpad=10)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.savefig('fig_2c_revision.pdf', dpi=400, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    x_list = ['Product', 'Product Titer', 'Gene', 'Temperature', 'pH', 'Carbon Source', 'Medium', 'Vessel']
    y_list = ['Qwen110b (D2Cell)', 'Qwen3 (D2Cell)', 'GPT4 (D2Cell)', 'Qwen110b', 'GPT4', 'Qwen3']
    d2cell_qwen = compare_d2cell('../../Result/RE Result/laser_d2cell_qwen.csv', '../../Data/RE Data/laser_dataset.csv')
    d2cell_qwen3 = compare_other('../../Result/RE Result/laser_d2cell_Qwen3.csv', '../../Data/RE Data/laser_dataset.csv')
    d2cell_gpt = compare_d2cell('../../Result/RE Result/laser_d2cell_gpt4.csv', '../../Data/RE Data/laser_dataset.csv')
    gpt = compare_other('../../Result/RE Result/laser_direct_gpt4.csv', '../../Data/RE Data/laser_dataset.csv')
    qwen = compare_other('../../Result/RE Result/laser_direct_qwen.csv', '../../Data/RE Data/laser_dataset.csv')
    qwen3 = compare_other('../../Result/RE Result/data_laser_qwen3.csv', '../../Data/RE Data/laser_dataset.csv')
    data_list = [d2cell_qwen, d2cell_qwen3, d2cell_gpt, qwen, gpt, qwen3]
    data_list = [[item * 100 for item in sublist] for sublist in data_list]
    heatmap(x_list=x_list, y_list=y_list, data=data_list)