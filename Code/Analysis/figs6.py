import matplotlib.pyplot as plt
import pandas as pd


def bar(ylabel,label_list,value_list, rotation=45, output=''):
    plt.figure(figsize=(3, 1.5), dpi=400)
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
    plt.bar([0.75, 1.5, 2.25], value_list, width=bar_width,
            color=['#f09b9b', '#fee090', '#74add1'], edgecolor='black', linewidth=0.5)
    plt.xticks([0.75, 1.5, 2.25], label_list, fontsize=7, rotation=rotation)
    plt.yticks(fontsize=7)
    plt.xlim(0, 3.2)
    plt.ylim(0, 150)

    plt.ylabel(ylabel, fontsize=7)
    plt.tight_layout()

    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def predict_target_number():
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_laser_predict.csv')
    df_ecoli_test = df_ecoli_test[df_ecoli_test['true label'] == 0]
    accuracy_multiple_gene = []
    accuracy_multiple_gene.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fseof_predict']]))
    accuracy_multiple_gene.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fvseof_predict']]))
    accuracy_multiple_gene.append(len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']]))
    return accuracy_multiple_gene



if __name__ == '__main__':
    xlabel = ['FSEOF', 'FVSEOF', 'D2Cell-pred']

    value_list = predict_target_number()
    bar(ylabel='Number of targets', label_list=xlabel, value_list=value_list,
        rotation=0, output='../../Result/figs6.pdf')