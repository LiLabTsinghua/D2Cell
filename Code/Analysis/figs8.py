import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def bar(ylabel,label_list,value_list, value_list2, value_list3, rotation=45, output='', x_label=''):

    # Setting General Parameters
    plt.figure(figsize=(2.75, 2), dpi=400)
    plt.rcParams.update({'font.size': 5})
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
            color=['#74add1'], label='$C. glutamicum$', edgecolor='black', linewidth=0.5)

    plt.bar([0.5, 1.4, 2.3], value_list2, width=bar_width,
            color=['#f09b9b'], label='$E. coli$', edgecolor='black', linewidth=0.5)

    plt.bar([0.7, 1.6, 2.5], value_list3, width=bar_width,
            color=['#fee090'], label='$S. cerevisiae$', edgecolor='black', linewidth=0.5)
    # plt.xticks([0.7, 1.6, 2.5], label_list, fontsize=6, rotation=rotation, fontstyle='italic')
    plt.xticks([0.7, 1.6, 2.5], label_list, fontsize=6, rotation=rotation)
    plt.yticks(fontsize=6)
    plt.xlim(0, 3.2)
    plt.ylim(0, 115)

    plt.ylabel(ylabel, fontsize=6)
    plt.xlabel(x_label, fontsize=6)
    plt.legend(loc='upper right', fontsize=5.5, frameon=False, labelspacing=0.25, handlelength=1.3, handletextpad=0.25)
    # plt.tight_layout()

    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()



def precision_multi_dataset():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_multiple_gene_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_multiple_gene_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_multiple_gene_result.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(precision_score(df_ecoli_test['true label'], df_ecoli_test['predict label'], pos_label=0))
    accuracy_test_dataset_d2cell.append(recall_score(df_ecoli_test['true label'], df_ecoli_test['predict label'], pos_label=0))
    accuracy_test_dataset_d2cell.append(f1_score(df_ecoli_test['true label'], df_ecoli_test['predict label'], pos_label=0))


    accuracy_test_dataset_fseof = []

    accuracy_test_dataset_fseof.append(
        precision_score(df_yeast_test['true label'], df_yeast_test['predict label'], pos_label=0))
    accuracy_test_dataset_fseof.append(
        recall_score(df_yeast_test['true label'], df_yeast_test['predict label'], pos_label=0))
    accuracy_test_dataset_fseof.append(
        f1_score(df_yeast_test['true label'], df_yeast_test['predict label'], pos_label=0))

    accuracy_test_dataset_fvseof = []
    accuracy_test_dataset_fvseof.append(
        precision_score(df_cg_test['true label'], df_cg_test['predict label'], pos_label=0))
    accuracy_test_dataset_fvseof.append(
        recall_score(df_cg_test['true label'], df_cg_test['predict label'], pos_label=0))
    accuracy_test_dataset_fvseof.append(
        f1_score(df_cg_test['true label'], df_cg_test['predict label'], pos_label=0))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    accuracy_test_dataset_fseof = [item * 100 for item in accuracy_test_dataset_fseof]
    accuracy_test_dataset_fvseof = [item * 100 for item in accuracy_test_dataset_fvseof]

    print(accuracy_test_dataset_d2cell)
    print(accuracy_test_dataset_fseof)
    return accuracy_test_dataset_d2cell, accuracy_test_dataset_fseof, accuracy_test_dataset_fvseof


if __name__ == '__main__':
    xlabel = ['Precision', 'Recall', 'F1-score']
    ecoli_test_result, yeast_test_result, cg_test_result = precision_multi_dataset()

    bar(ylabel='Value (%)', label_list=xlabel, value_list=cg_test_result, value_list2=ecoli_test_result,
        value_list3=yeast_test_result,
        rotation=0, output='../../Result/figs8.pdf', x_label='Multiple gene modification test set')