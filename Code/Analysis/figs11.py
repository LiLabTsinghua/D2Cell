import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import FuncFormatter
import numpy as np

def bar(ylabel,label_list,value_list, rotation=45, output=''):

    # Setting General Parameters
    plt.figure(figsize=(4, 2), dpi=400)
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
    colors = ['#bdbdbd', '#bdbdbd'] + ['#74add1'] * (len(value_list) - 2)
    plt.bar([0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3], value_list, width=bar_width,
            color=colors, edgecolor='black', linewidth=0.5)
    plt.xticks([0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3], label_list, fontsize=7, rotation=45, ha='right')
    plt.yticks(fontsize=7)
    plt.xlim(0, 3.6)
    # plt.ylim(0, 65)

    plt.ylabel(ylabel, fontsize=7)
    # plt.tight_layout()

    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


def accuracy_test_dataset():
    df_ecoli_test = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_zero_shot.csv')
    df_ecoli_test_few_50 = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_few_shot_50.csv')
    df_ecoli_test_few_100 = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_few_shot_100.csv')
    df_ecoli_test_few_200 = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_few_shot_200.csv')
    df_ecoli_test_few_500 = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_few_shot_500.csv')

    accuracy_test_dataset_d2cell = []
    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fseof_predict']]) / len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['fvseof_predict']]) / len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test[df_ecoli_test['true label'] == df_ecoli_test['predict label']]) / len(df_ecoli_test))
    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test_few_50[df_ecoli_test_few_50['true label'] == df_ecoli_test_few_50['predict label']]) / len(df_ecoli_test_few_50))

    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test_few_100[df_ecoli_test_few_50['true label'] == df_ecoli_test_few_100['predict label']]) / len(
            df_ecoli_test_few_100))
    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test_few_200[df_ecoli_test_few_50['true label'] == df_ecoli_test_few_200['predict label']]) / len(
            df_ecoli_test_few_200))
    accuracy_test_dataset_d2cell.append(
        len(df_ecoli_test_few_50[df_ecoli_test_few_500['true label'] == df_ecoli_test_few_500['predict label']]) / len(
            df_ecoli_test_few_500))

    accuracy_test_dataset_d2cell = [item * 100 for item in accuracy_test_dataset_d2cell]
    return accuracy_test_dataset_d2cell


if __name__ == '__main__':
    xlabel = ['FSEOF', 'FVSEOF', 'Zero shot', 'Few shot (50)', 'Few shot (100)', 'Few shot (200)', 'Few shot(500)']

    D2Cell_test_result = accuracy_test_dataset()
    print(D2Cell_test_result)
    bar(ylabel='Accuracy on test set (%)', label_list=xlabel, value_list=D2Cell_test_result,
        rotation=45, output='../../Result/figs11.pdf')