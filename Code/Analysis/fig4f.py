import matplotlib.pyplot as plt
import pandas as pd


def bar(ylabel,label_list,value_list, rotation=45, output=''):

    # Setting General Parameters
    plt.figure(figsize=(2.75, 1.5), dpi=400)
    plt.rcParams.update({'font.size':7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    bar_width = 0.2
    plt.bar([0.5, 1.5, 2.5], value_list, width=bar_width,
            color=['#74add1'], edgecolor='black', linewidth=0.5)
    plt.xticks([0.5, 1.5, 2.5], label_list, fontsize=7, rotation=rotation, fontstyle='italic')
    plt.yticks(fontsize=7)
    plt.xlim(0, 3.2)
    plt.ylim(40, 105)

    plt.ylabel(ylabel, fontsize=7)
    # plt.tight_layout()

    # plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def accuracy_unseen_product():
    df_cg_test = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_multiple_gene_result.csv')
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_multiple_gene_result.csv')
    df_yeast_test = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_multiple_gene_result.csv')

    accuracy_multiple_gene = []
    accuracy_multiple_gene.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']])/len(df_ecoli_test))
    accuracy_multiple_gene.append(
        len(df_yeast_test[df_yeast_test['true label'] == df_yeast_test['predict label']]) / len(df_yeast_test))
    accuracy_multiple_gene.append(
        len(df_cg_test[df_cg_test['true label'] == df_cg_test['predict label']]) / len(df_cg_test))

    accuracy_multiple_gene = [item * 100 for item in accuracy_multiple_gene]
    return accuracy_multiple_gene


if __name__ == '__main__':
    xlabel = ['E. coli', 'S. cerevisiae', 'C. glutamicum']
    value_list = accuracy_unseen_product()
    bar(ylabel='Accuracy on multiple gene\nmodification (%)', label_list=xlabel, value_list=value_list,
        rotation=0, output='../../Result/fig4_f.pdf')