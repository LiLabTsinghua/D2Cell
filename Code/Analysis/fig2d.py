import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_dual_kernel_density(data1, data2, data3, figure_path, max_ylim):
    # Setting General Parameters
    plt.figure(figsize=(2.75, 1.5), dpi=400)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['font.family'] = 'Arial'

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)

    sns.kdeplot(data2, fill=True, color='#225ea8', label='Cellulosic ethanol')
    sns.kdeplot(data1, fill=True, color='#41ab5d', label='Monoterpene')
    sns.kdeplot(data3, fill=True, color='#ce1256', label='Protein')

    plt.xlabel('Accuracy of data extraction', fontsize=8)
    plt.ylabel('Density', fontsize=8)
    legend = plt.legend(frameon=False, fontsize=7, ncol=3, handletextpad=0.2, handlelength=1, borderaxespad=0)

    plt.ylim(0,max_ylim)
    plt.xlim(-0.3,1.5)

    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    plt.savefig(figure_path, dpi=400, bbox_inches='tight')
    plt.show()


def check_result(df):
    df = df.dropna(how='all')
    df = df[df['knock out gene'].notnull() | df['overexpress gene'].notnull() | df[
        'heterologous gene'].notnull()]
    data_list = []
    paper_list = sorted(list(set(df['doi'].tolist())))
    for paper_id in paper_list:
        df_paper = df[df['doi'] == paper_id]
        data_list.append(len(df_paper[df_paper['Check'] == 'yes']) / len(df_paper))
    return data_list


if __name__ == '__main__':
    df_mono = pd.read_csv('../../Result/RE Result/product/monoterpene_check.csv')
    mono_data_list = check_result(df_mono)
    df_cellulosic = pd.read_csv('../../Result/RE Result/product/cellulosic_ethanol_check.csv')
    cellulosic_data_list = check_result(df_cellulosic)
    df_prot = pd.read_csv('../../Result/RE Result/product/protein_check.csv')
    prot_data_list = check_result(df_prot)
    plot_dual_kernel_density(mono_data_list, cellulosic_data_list, prot_data_list, '../../Result/fig2_d.pdf', 8)