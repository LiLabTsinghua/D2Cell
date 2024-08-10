import matplotlib.pyplot as plt
import pandas as pd
import shutil
import matplotlib
import numpy as np
import re
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def capitalize_first_letter(word):
    try:
        match = re.search(r'[a-zA-Z]', word)
        if match:
            index = match.start()
            return word[:index] + word[index].upper() + word[index+1:]
        return word
    except TypeError:
        return word


def plot_heatmap(data1, data2, data3, labels, strain_type='$\t{E. coli}$', output_path='../article_figure/figs4_c.pdf',
                 sorted_index=[], figsize=9):
    data_matrix = np.array([data1, data2, data3])

    plt.figure(figsize=(figsize, 1.5), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    ax = sns.heatmap(data_matrix, annot=False, cmap='Blues', xticklabels=labels, vmin=0.3,
                     yticklabels=['D2Cell-pred', 'FSEOF', 'FVSEOF'], cbar_kws={'label': 'Accuracy on test data', 'pad': 0.01})
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=1.5)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(['D2Cell-pred', 'FSEOF', 'FVSEOF'], fontsize=6, rotation=45)
    ax.set_xlabel(strain_type + ' product', fontsize=6)
    ax.set_ylabel('', fontsize=7)

    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.show()
    return sorted_index


if __name__ == '__main__':
    shutil.rmtree(matplotlib.get_cachedir())

    df = pd.read_csv('../../Result/D2Cell-pred Result/Cg/cg_test_dataset.csv')
    df['name'] = df['name'].apply(capitalize_first_letter)
    labels = (list(set(df['name'].tolist())))
    labels = sorted([label for label in labels if isinstance(label, str)])
    our_acc = []
    FSEOF_acc = []
    FVSEOF_acc = []
    for label in labels:
        df_label = df[df['name'] == label]
        our_acc.append((df_label['inf_label_01'] == df_label['predict label']).sum()/len(df_label))
        try:
            FSEOF_acc.append((df_label['inf_label_01'] == df_label['fseof_predict']).sum()/len(df_label))
        except AttributeError:
            FSEOF_acc.append(0)
        try:
            FVSEOF_acc.append((df_label['inf_label_01'] == df_label['fvseof_predict']).sum() / len(df_label))
        except AttributeError:
            FVSEOF_acc.append(0)
    sorted_index = plot_heatmap(our_acc, FSEOF_acc, FVSEOF_acc, labels, strain_type='$\t{C. glutamicum}$',
                            output_path='../../Result/figs5_c.pdf', figsize=3)


    df = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/ecoli_test_dataset.csv')
    df['name'] = df['name'].apply(capitalize_first_letter)
    labels = (list(set(df['name'].tolist())))
    labels = sorted([label for label in labels if isinstance(label, str)])
    our_acc = []
    FSEOF_acc = []
    FVSEOF_acc = []
    for label in labels:
        df_label = df[df['name'] == label]
        our_acc.append((df_label['inf_label_01'] == df_label['predict label']).sum() / len(df_label))
        try:
            FSEOF_acc.append((df_label['inf_label_01'] == df_label['fseof_predict']).sum() / len(df_label))
        except AttributeError:
            FSEOF_acc.append(0)
        try:
            FVSEOF_acc.append((df_label['inf_label_01'] == df_label['fvseof_predict']).sum() / len(df_label))
        except AttributeError:
            FVSEOF_acc.append(0)
    sorted_index = plot_heatmap(our_acc, FSEOF_acc, FVSEOF_acc, labels, strain_type='$\t{E. coli}$',
                            output_path='../../Result/figs5_a.pdf')


    df = pd.read_csv('../../Result/D2Cell-pred Result/Yeast/yeast_test_dataset.csv')
    df['name'] = df['name'].apply(capitalize_first_letter)
    labels = (list(set(df['name'].tolist())))
    labels = sorted([label for label in labels if isinstance(label, str)])
    our_acc = []
    FSEOF_acc = []
    FVSEOF_acc = []
    for label in labels:
        df_label = df[df['name'] == label]
        our_acc.append((df_label['inf_label_01'] == df_label['predict label']).sum()/len(df_label))
        try:
            FSEOF_acc.append((df_label['inf_label_01'] == df_label['fseof_predict']).sum()/len(df_label))
        except AttributeError:
            FSEOF_acc.append(0)
        try:
            FVSEOF_acc.append((df_label['inf_label_01'] == df_label['fvseof_predict']).sum() / len(df_label))
        except AttributeError:
            FVSEOF_acc.append(0)
    sorted_index = plot_heatmap(our_acc, FSEOF_acc, FVSEOF_acc, labels, strain_type='$\t{S. cerevisiae}$',
                            output_path='../../Result/figs5_b.pdf',figsize=5.5)
