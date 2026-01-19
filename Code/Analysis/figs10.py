import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import FuncFormatter
import numpy as np

def bar(ylabel,label_list,value_list, value_list2, value_list3, rotation=45, output=''):

    # Setting General Parameters
    plt.figure(figsize=(2, 1.5), dpi=400)
    plt.rcParams.update({'font.size': 6})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)

    bar_width = 0.2
    plt.bar([0.5, 1.4, 2.3], value_list, width=bar_width,
            color=['#f09b9b'], label='wo simulated data', edgecolor='black', linewidth=0.5)

    plt.bar([0.7, 1.6, 2.5], value_list2, width=bar_width,
            color=['#fee090'], label='Half Simulated data', edgecolor='black', linewidth=0.5)

    plt.bar([0.9, 1.8, 2.7], value_list3, width=bar_width,
            color=['#74add1'], label='Simulated data', edgecolor='black', linewidth=0.5)
    plt.xticks([0.7, 1.6, 2.5], label_list, fontsize=7, rotation=rotation, fontstyle='italic')
    plt.yticks(fontsize=6)
    plt.xlim(0, 3.2)
    plt.ylim(40, 110)

    plt.ylabel(ylabel, fontsize=6)
    plt.legend(loc='upper left', fontsize=5.5, frameon=False, labelspacing=0.25, handlelength=1.3, handletextpad=0.25)
    # plt.tight_layout()

    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def accuracy_test_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result_sim0.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result_sim0.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result_sim0.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_d2cell.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))


    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result_sim50.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result_sim50.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result_sim50.csv')
    accuracy_test_dataset_fseof = []
    accuracy_test_dataset_fseof.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']]) / len(df_ecoli_test))
    accuracy_test_dataset_fseof.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_fseof.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))


    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result_sim100.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result_sim100.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result_sim100.csv')
    accuracy_test_dataset_fvseof = []
    accuracy_test_dataset_fvseof.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']]) / len(df_ecoli_test))
    accuracy_test_dataset_fvseof.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_fvseof.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_fseof = [item * 100 for item in accuracy_test_dataset_fseof]
    accuracy_test_dataset_fvseof = [item * 100 for item in accuracy_test_dataset_fvseof]
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_fseof, accuracy_test_dataset_fvseof


if __name__ == '__main__':
    xlabel = ['E. coli', 'S. cerevisiae', 'C. glutamicum']

    D2Cell_test_result, FSEOF_test_result, FVSEOF_test_result = accuracy_test_dataset()
    print(D2Cell_test_result, FSEOF_test_result, FVSEOF_test_result)
    bar(ylabel='Accuracy on test dataset (%)', label_list=xlabel, value_list=D2Cell_test_result, value_list2=FSEOF_test_result, value_list3=FVSEOF_test_result,
        rotation=45, output='../../Result/figs10.pdf')