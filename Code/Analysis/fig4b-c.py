import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import FuncFormatter
import numpy as np
from sklearn.metrics import precision_score

def bar(ylabel,label_list,value_list, value_list2, value_list3, rotation=45, output=''):

    # Setting General Parameters
    plt.figure(figsize=(2.75, 1.5), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)

    bar_width = 0.2
    plt.bar([0.9, 1.8, 2.7], value_list, width=bar_width,
            color=['#74add1'], label='D2Cell-pred', edgecolor='black', linewidth=0.5)

    plt.bar([0.5, 1.4, 2.3], value_list2, width=bar_width,
            color=['#f09b9b'], label='FSEOF', edgecolor='black', linewidth=0.5)

    plt.bar([0.7, 1.6, 2.5], value_list3, width=bar_width,
            color=['#fee090'], label='FVSEOF', edgecolor='black', linewidth=0.5)
    plt.xticks([0.7, 1.6, 2.5], label_list, fontsize=7, rotation=rotation, fontstyle='italic')
    plt.yticks(fontsize=7)
    plt.xlim(0, 3.2)
    plt.ylim(40, 105)

    plt.ylabel(ylabel, fontsize=7)
    plt.legend(loc='upper left', fontsize=6, frameon=False, labelspacing=0.25, handlelength=1.3, handletextpad=0.25)
    # plt.tight_layout()

    # plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def accuracy_test_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_d2cell.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_test_dataset_fseof = []
    accuracy_test_dataset_fseof.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fseof_predict']]) / len(df_ecoli_test))
    accuracy_test_dataset_fseof.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['fseof_predict']]) / len(df_yeast_test))
    accuracy_test_dataset_fseof.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['fseof_predict']]) / len(df_cg_test))

    accuracy_test_dataset_fvseof = []
    accuracy_test_dataset_fvseof.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fvseof_predict']]) / len(df_ecoli_test))
    accuracy_test_dataset_fvseof.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['fvseof_predict']]) / len(df_yeast_test))
    accuracy_test_dataset_fvseof.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['fvseof_predict']]) / len(df_cg_test))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_fseof = [item * 100 for item in accuracy_test_dataset_fseof]
    accuracy_test_dataset_fvseof = [item * 100 for item in accuracy_test_dataset_fvseof]
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_fseof, accuracy_test_dataset_fvseof

def precision_test_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_result.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['predict label'], pos_label=0))
    accuracy_test_dataset_d2cell.append(precision_score(df_yeast_test['true label'], df_yeast_test['predict label'], pos_label=0))
    accuracy_test_dataset_d2cell.append(precision_score(df_cg_test['true label'], df_cg_test['predict label'], pos_label=0))

    accuracy_test_dataset_fseof = []
    accuracy_test_dataset_fseof.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['fseof_predict'], pos_label=0))
    accuracy_test_dataset_fseof.append(precision_score(df_yeast_test['true label'], df_yeast_test['fseof_predict'], pos_label=0))
    accuracy_test_dataset_fseof.append(precision_score(df_cg_test['true label'], df_cg_test['fseof_predict'], pos_label=0))

    accuracy_test_dataset_fvseof = []
    accuracy_test_dataset_fvseof.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['fvseof_predict'], pos_label=0))
    accuracy_test_dataset_fvseof.append(precision_score(df_yeast_test['true label'], df_yeast_test['fvseof_predict'], pos_label=0))
    accuracy_test_dataset_fvseof.append(precision_score(df_cg_test['true label'], df_cg_test['fvseof_predict'], pos_label=0))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_fseof = [item * 100 for item in accuracy_test_dataset_fseof]
    accuracy_test_dataset_fvseof = [item * 100 for item in accuracy_test_dataset_fvseof]
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_fseof, accuracy_test_dataset_fvseof

def accuracy_unseen_product():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_unseen_product_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_unseen_product_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_unseen_product_result.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_test_dataset_d2cell.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_test_dataset_fseof = []
    accuracy_test_dataset_fseof.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fseof_predict']]) / len(df_ecoli_test))
    accuracy_test_dataset_fseof.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['fseof_predict']]) / len(df_yeast_test))
    accuracy_test_dataset_fseof.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['fseof_predict']]) / len(df_cg_test))

    accuracy_test_dataset_fvseof = []
    accuracy_test_dataset_fvseof.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fvseof_predict']]) / len(df_ecoli_test))
    accuracy_test_dataset_fvseof.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['fvseof_predict']]) / len(df_yeast_test))
    accuracy_test_dataset_fvseof.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['fvseof_predict']]) / len(df_cg_test))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_fseof = [item * 100 for item in accuracy_test_dataset_fseof]
    accuracy_test_dataset_fvseof = [item * 100 for item in accuracy_test_dataset_fvseof]
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_fseof, accuracy_test_dataset_fvseof

def precision_unseen_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_unseen_product_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_unseen_product_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_unseen_product_result.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['predict label'], pos_label=0))
    accuracy_test_dataset_d2cell.append(precision_score(df_yeast_test['true label'], df_yeast_test['predict label'], pos_label=0))
    accuracy_test_dataset_d2cell.append(precision_score(df_cg_test['true label'], df_cg_test['predict label'], pos_label=0))

    accuracy_test_dataset_fseof = []
    accuracy_test_dataset_fseof.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['fseof_predict'], pos_label=0))
    accuracy_test_dataset_fseof.append(precision_score(df_yeast_test['true label'], df_yeast_test['fseof_predict'], pos_label=0))
    accuracy_test_dataset_fseof.append(precision_score(df_cg_test['true label'], df_cg_test['fseof_predict'], pos_label=0))

    accuracy_test_dataset_fvseof = []
    accuracy_test_dataset_fvseof.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['fvseof_predict'], pos_label=0))
    accuracy_test_dataset_fvseof.append(precision_score(df_yeast_test['true label'], df_yeast_test['fvseof_predict'], pos_label=0))
    accuracy_test_dataset_fvseof.append(precision_score(df_cg_test['true label'], df_cg_test['fvseof_predict'], pos_label=0))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_fseof = [item * 100 for item in accuracy_test_dataset_fseof]
    accuracy_test_dataset_fvseof = [item * 100 for item in accuracy_test_dataset_fvseof]
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_fseof, accuracy_test_dataset_fvseof


if __name__ == '__main__':
    xlabel = ['E. coli', 'S. cerevisiae', 'C. glutamicum']

    D2Cell_test_result, FSEOF_test_result, FVSEOF_test_result = accuracy_test_dataset()
    bar(ylabel='Accuracy on test dataset (%)', label_list=xlabel, value_list=D2Cell_test_result, value_list2=FSEOF_test_result, value_list3=FVSEOF_test_result,
        rotation=0, output='../../Result/fig4_b.pdf')

    # D2Cell_test_result, FSEOF_test_result, FVSEOF_test_result = accuracy_unseen_product()
    # bar(ylabel='Accuracy on test dataset with\nout-of-distribution product (%)', label_list=xlabel, value_list=D2Cell_test_result, value_list2=FSEOF_test_result, value_list3=FVSEOF_test_result,
    #     rotation=0, output='../../Result/fig4_c.pdf')

    # D2Cell_test_result, FSEOF_test_result, FVSEOF_test_result = precision_test_dataset()
    # bar(ylabel='Precision on test dataset (%)', label_list=xlabel, value_list=D2Cell_test_result,
    #     value_list2=FSEOF_test_result, value_list3=FVSEOF_test_result,
    #     rotation=0, output='../../Result/fig4_b.pdf')

    # D2Cell_test_result, FSEOF_test_result, FVSEOF_test_result = precision_unseen_dataset()
    # bar(ylabel='Precision on test dataset with\nout-of-distribution product (%)', label_list=xlabel, value_list=D2Cell_test_result, value_list2=FSEOF_test_result, value_list3=FVSEOF_test_result,
    #     rotation=0, output='../../Result/fig4_c.pdf')