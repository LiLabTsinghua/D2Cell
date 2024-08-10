import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个示例DataFrame
import pandas as pd
import numpy as np


def draw_boxplot(data_list):
    np.random.seed(10)
    plt.figure(figsize=(2.25, 1.5), dpi=400)
    plt.rcParams['pdf.fonttype'] = 42
    plt.subplots_adjust(left=0.2, right=0.95, top=0.75, bottom=0.1)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 8})
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    colors = ['#edf8b1', '#74a9cf', '#edf8b1', '#74a9cf']
    box_colors = ['black','black','black','black']
    whiskers_colors = ['black','black', 'black','black']
    boxplot = plt.boxplot(data_list, patch_artist=True, boxprops=dict(facecolor='#b3cde3', color='black'),
                medianprops=dict(color='#bd0026', linewidth=1), whiskerprops=dict(color='black', linewidth=0.5),
                capprops=dict(color='black', linewidth=0.5), flierprops=dict(marker=None))
    for patch, color, color2 in zip(boxplot['boxes'], colors, box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color2)
        patch.set_linewidth(0.5)
        patch.set_alpha(1)

    for whisker, color in zip(boxplot['whiskers'], whiskers_colors):
        whisker.set_color(color)
        whisker.set_linewidth(0.5)
        whisker.set_alpha(1)

    for cap, color in zip(boxplot['caps'], whiskers_colors):
        cap.set_color(color)
        cap.set_linewidth(0.5)
        cap.set_alpha(1)

    for i, d in enumerate(data_list):
        y = np.random.normal(i + 1, 0.04, size=len(d))
        plt.scatter(y, d, alpha=0.75, color='gray', linewidths=0, zorder=2, s=7.5)
    plt.xticks([1, 2, 3, 4], ['Qwen', 'GPT4', 'Qwen (D2Cell)', 'GPT4 (D2Cell)'], fontsize=8
               , rotation=45, ha='right')
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    plt.ylabel('Accuracy of data extraction (%)', fontsize=8)
    plt.savefig('../../Result/figs1_c.pdf', dpi=400, bbox_inches='tight')
    plt.show()


def compare_data(df1, df2, df3, df4):
    doi_list = sorted(list(set([x for x in df1['doi'].tolist() if isinstance(x, str)])))
    gpt4_list = []
    qwen_list = []
    gpt4_direct_list = []
    qwen_direct_list = []
    for doi in doi_list:
        df1_current = df1[df1['doi'] == doi]
        df1_current = df1_current.drop_duplicates(subset='product titer', keep='first')
        df2_current = df2[df2['doi'] == doi]
        df2_current = df2_current.drop_duplicates(subset='product titer', keep='first')
        df3_current = df3[df3['doi'] == doi]
        df3_current = df3_current.drop_duplicates(subset='product titer', keep='first')
        df4_current = df4[df4['doi'] == doi]
        df4_current = df4_current.drop_duplicates(subset='product titer', keep='first')
        try:
            gpt4_list.append(len(df1_current[df1_current['check'] == 'yes']) / len(df1_current))
        except ZeroDivisionError:
            gpt4_list.append(0)
        try:
            qwen_list.append(len(df2_current[df2_current['check'] == 'yes']) / len(df2_current))
        except ZeroDivisionError as e:
            qwen_list.append(0)
        try:
            gpt4_direct_list.append(len(df3_current[df3_current['check'] == 'yes']) / len(df3_current))
        except ZeroDivisionError:
            gpt4_direct_list.append(0)
        try:
            qwen_direct_list.append(len(df4_current[df4_current['check'] == 'yes']) / len(df4_current))
        except ZeroDivisionError as e:
            qwen_direct_list.append(0)
    return gpt4_list, qwen_list, gpt4_direct_list, qwen_direct_list


if __name__ == '__main__':
    df1 = pd.read_csv('../../Result/RE Result/GPT4_D2Cell_100_paper_result.csv')
    df2 = pd.read_csv('../../Result/RE Result/Qwen_D2Cell_100_paper_result.csv')
    df3 = pd.read_csv('../../Result/RE Result/GPT4_Direct_100_paper_result.csv')
    df4 = pd.read_csv('../../Result/RE Result/Qwen_Direct_100_paper_result.csv')
    gpt_data, qwen_data, gpt4_direct_list, qwen_direct_list = compare_data(df1, df2, df3, df4)
    data_list = [qwen_direct_list, gpt4_direct_list, qwen_data, gpt_data]
    data_list = [[value * 100 for value in sublist] for sublist in data_list]
    draw_boxplot(data_list)