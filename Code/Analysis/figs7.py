import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import FuncFormatter
import numpy as np

def bar(ylabel,label_list,value_list, value_list2, rotation=45, output='', strain='E. coli'):

    # Setting General Parameters
    plt.figure(figsize=(1.5, 1.5), dpi=400)
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)

    bar_width = 0.1
    plt.bar([1, 1.3, 1.6], value_list, width=bar_width,
            color=['#74add1'], label='D2Cell-pred', edgecolor='black', linewidth=0.5)

    plt.bar([0.9, 1.2, 1.5], value_list2, width=bar_width,
            color=['#f09b9b'], label='No GEMs feature', edgecolor='black', linewidth=0.5)

    plt.xticks([0.95, 1.25, 1.55], label_list, fontsize=8, rotation=rotation, ha='right')
    plt.yticks(fontsize=8)
    plt.xlim(0.7,1.8)
    plt.ylim(60, 100)
    plt.xlabel(strain, fontsize=8, fontstyle='italic')
    plt.ylabel(ylabel, fontsize=8)
    plt.legend(loc='upper left', fontsize=6, frameon=False, labelspacing=0.25, handlelength=1.3, handletextpad=0.25)
    # plt.tight_layout()

    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def accuracy_test_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result.csv')
    df_cg_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result_no_gem.csv')
    df_ecoli_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result_no_gem.csv')
    df_yeast_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result_no_gem.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_d2cell.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_test_dataset_no_gem = []
    accuracy_test_dataset_no_gem.append(
        len(df_ecoli_test_no_gem[df_ecoli_test_no_gem['true label'] == df_ecoli_test_no_gem['predict label']]) / len(df_ecoli_test_no_gem ))
    accuracy_test_dataset_no_gem.append(
        len(df_yeast_test_no_gem[df_yeast_test_no_gem['true label'] == df_yeast_test_no_gem['predict label']]) / len(df_yeast_test_no_gem))
    accuracy_test_dataset_no_gem.append(
        len(df_cg_test_no_gem[df_cg_test_no_gem['true label'] == df_cg_test_no_gem['predict label']]) / len(df_cg_test_no_gem))

    print(accuracy_test_dataset_d2cell)
    print(accuracy_test_dataset_no_gem)
    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_no_gem = [item * 100 for item in accuracy_test_dataset_no_gem]
    # accuracy_test_dataset_d2cell = sum([item * 100 for item in accuracy_test_dataset_d2cell])/3
    # accuracy_test_dataset_no_gem = sum([item * 100 for item in accuracy_test_dataset_no_gem])/3
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_no_gem


def accuracy_unseen_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_unseen_product_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_unseen_product_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_unseen_product_result.csv')
    df_cg_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_unseen_product_no_gem.csv')
    df_ecoli_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_unseen_product_no_gem.csv')
    df_yeast_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_unseen_product_no_gem.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_d2cell.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_test_dataset_no_gem = []
    accuracy_test_dataset_no_gem.append(
        len(df_ecoli_test_no_gem [df_ecoli_test_no_gem['true label'] == df_ecoli_test_no_gem['predict label']]) / len(df_ecoli_test_no_gem ))
    accuracy_test_dataset_no_gem.append(
        len(df_yeast_test_no_gem[df_yeast_test_no_gem['true label'] == df_yeast_test_no_gem['predict label']]) / len(df_yeast_test_no_gem))
    accuracy_test_dataset_no_gem.append(
        len(df_cg_test_no_gem[df_cg_test_no_gem['true label'] == df_cg_test_no_gem['predict label']]) / len(df_cg_test_no_gem))

    print('unseen d2cell',accuracy_test_dataset_d2cell)
    print('unseen no gem',accuracy_test_dataset_no_gem)
    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_no_gem = [item * 100 for item in accuracy_test_dataset_no_gem]
    # accuracy_test_dataset_d2cell = sum([item * 100 for item in accuracy_test_dataset_d2cell])/3
    # accuracy_test_dataset_no_gem = sum([item * 100 for item in accuracy_test_dataset_no_gem])/3
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_no_gem


def accuracy_mutiple_gene_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_multiple_gene_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_multiple_gene_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_multiple_gene_result.csv')
    df_cg_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_multiple_gene_result_no_gem.csv')
    df_ecoli_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_multiple_gene_result_no_gem.csv')
    df_yeast_test_no_gem = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_multiple_gene_result_no_gem.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_d2cell.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_test_dataset_no_gem = []
    accuracy_test_dataset_no_gem.append(
        len(df_ecoli_test_no_gem [df_ecoli_test_no_gem['true label'] == df_ecoli_test_no_gem['predict label']]) / len(df_ecoli_test_no_gem ))
    accuracy_test_dataset_no_gem.append(
        len(df_yeast_test_no_gem[df_yeast_test_no_gem['true label'] == df_yeast_test_no_gem['predict label']]) / len(df_yeast_test_no_gem))
    accuracy_test_dataset_no_gem.append(
        len(df_cg_test_no_gem[df_cg_test_no_gem['true label'] == df_cg_test_no_gem['predict label']]) / len(df_cg_test_no_gem))

    print('unseen d2cell',accuracy_test_dataset_d2cell)
    print('unseen no gem',accuracy_test_dataset_no_gem)
    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_no_gem = [item * 100 for item in accuracy_test_dataset_no_gem]
    # accuracy_test_dataset_d2cell = sum([item * 100 for item in accuracy_test_dataset_d2cell])/3
    # accuracy_test_dataset_no_gem = sum([item * 100 for item in accuracy_test_dataset_no_gem])/3
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_no_gem


if __name__ == '__main__':
    xlabel = ['Test data', 'Out-of-distribution product set', 'Mutiple gene test set']

    D2Cell_test_result, no_gem_test_result = accuracy_test_dataset()
    D2Cell_unseen_result, no_gem_unseen_result = accuracy_unseen_dataset()
    D2Cell_muti_gene, no_gem_muti_gene = accuracy_mutiple_gene_dataset()
    d2cell_result = [D2Cell_test_result[0], D2Cell_unseen_result[0], D2Cell_muti_gene[0]]
    no_gem_result = [no_gem_test_result[0], no_gem_unseen_result[0], no_gem_muti_gene[0]]
    bar(ylabel='Accuracy on test dataset (%)', label_list=xlabel, value_list=d2cell_result, value_list2=no_gem_result,
        rotation=45, output='../../Result/figs7_a.pdf', strain='E. coli')

    d2cell_result = [D2Cell_test_result[1], D2Cell_unseen_result[1], D2Cell_muti_gene[1]]
    no_gem_result = [no_gem_test_result[1], no_gem_unseen_result[1], no_gem_muti_gene[1]]
    bar(ylabel='Accuracy on test dataset (%)', label_list=xlabel, value_list=d2cell_result, value_list2=no_gem_result,
        rotation=45, output='../../Result/figs7_b.pdf', strain='S. cerevisiae')

    d2cell_result = [D2Cell_test_result[2], D2Cell_unseen_result[2], D2Cell_muti_gene[2]]
    no_gem_result = [no_gem_test_result[2], no_gem_unseen_result[2], no_gem_muti_gene[2]]
    bar(ylabel='Accuracy on test dataset (%)', label_list=xlabel, value_list=d2cell_result, value_list2=no_gem_result,
        rotation=45, output='../../Result/figs7_c.pdf', strain='C. glutamicum')
