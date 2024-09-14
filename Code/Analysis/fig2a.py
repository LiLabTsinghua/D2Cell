import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar(ylabel, label_list, value_list, number_list, rotation=45, output='', ylim=[60, 85]):

    # Setting General Parameters
    plt.figure(figsize=(3, 1.5), dpi=400)
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42

    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['right'].set_linewidth(0.5)
    bar_width = 0.2
    index = np.arange(len(label_list))
    colors = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#41b6c4', '#225ea8']
    ax1.bar(index, value_list, width=bar_width, color=colors, linewidth=0.5, edgecolor='black')

    ax1.set_ylabel(ylabel, fontsize=8)
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    ax1.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    ax1.set_xticks(index)
    ax1.set_xticklabels(label_list, fontsize=8, rotation=rotation, ha='right')

    # Creating secondary axes
    ax2 = ax1.twinx()
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    ax2.plot(index, number_list, color='#ce1256', marker='o', markersize=3, linewidth=0.5)
    ax2.set_ylim(200, 1000)
    ax2.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5, colors='#ce1256')
    ax2.set_ylabel('Data extraction number', fontsize=8)
    # plt.tight_layout()
    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def compare_data(df1, df2, df3, df4):
    doi_list = sorted(list(set([x for x in df1['doi'].tolist() if isinstance(x, str)])))
    gpt_sum = 0
    qwen_sum = 0
    gpt_direct_sum = 0
    qwen_direct_sum = 0
    gpt_all = 0
    qwen_all = 0
    gpt_direct_all = 0
    qwen_direct_all = 0
    for doi in doi_list:
        df1_current = df1[df1['doi'] == doi]
        df1_current = df1_current.drop_duplicates(subset='product titer', keep='first')
        df2_current = df2[df2['doi'] == doi]
        df2_current = df2_current.drop_duplicates(subset='product titer', keep='first')
        df3_current = df3[df3['doi'] == doi]
        df3_current = df3_current.drop_duplicates(subset='product titer', keep='first')
        df4_current = df4[df4['doi'] == doi]
        df4_current = df4_current.drop_duplicates(subset='product titer', keep='first')
        gpt_sum += len(df1_current[df1_current['check'] == 'yes'])
        qwen_sum += len(df2_current[df2_current['check'] == 'yes'])
        gpt_direct_sum += len(df3_current[df3_current['check'] == 'yes'])
        qwen_direct_sum += len(df4_current[df4_current['check'] == 'yes'])
        gpt_all += len(df1_current)
        qwen_all += len(df2_current)
        gpt_direct_all += len(df3_current)
        qwen_direct_all += len(df4_current)
    print('all gpt', gpt_sum/gpt_all)
    print('all qwen', qwen_sum/qwen_all)
    print('direct gpt', gpt_direct_sum/gpt_direct_all)
    print('direct qwen', qwen_direct_sum/qwen_direct_all)
    return ([qwen_direct_sum/qwen_direct_all, gpt_direct_sum/gpt_direct_all, qwen_sum/qwen_all, gpt_sum/gpt_all],
            [qwen_direct_all, gpt_direct_all, qwen_all, gpt_all])


if __name__ == '__main__':
    df1 = pd.read_csv('../../Result/RE Result/GPT4_D2Cell_100_paper_result.csv')
    df2 = pd.read_csv('../../Result/RE Result/Qwen_D2Cell_100_paper_result.csv')
    df3 = pd.read_csv('../../Result/RE Result/GPT4_Direct_100_paper_result.csv')
    df4 = pd.read_csv('../../Result/RE Result/Qwen_Direct_100_paper_result.csv')
    value_list, number_list = compare_data(df1, df2, df3, df4)
    value_list = [value * 100 for value in value_list]
    xlabel = ['Qwen', 'GPT4', 'Qwen (D2Cell)', 'GPT4 (D2Cell)']
    bar(ylabel='Accuracy of data extraction (%)', label_list=xlabel, value_list=value_list, number_list=number_list,
        rotation=45, output='../../Result/fig2_a.pdf')