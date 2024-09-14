import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted
from matplotlib_venn import venn3
import pandas as pd
from collections import Counter

def venn3_plot(csv_file, output='../../Result/fig4_d1.pdf'):
    plt.figure(figsize=(2, 1.1), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    df = pd.read_csv(csv_file)
    df = df[df['inf_label_01'] == 0]
    fseof = df['fseof_predict'].tolist()
    fvseof = df['fvseof_predict'].tolist()
    d2cell = df['predict label'].tolist()
    counts = Counter(zip(fseof, fvseof, d2cell))

    only_fseof = counts[(0, 1, 1)]
    only_fvseof = counts[(1, 0, 1)]
    only_d2cell = counts[(1, 1, 0)]

    fseof_fvseof = counts[(0, 0, 1)]
    fseof_d2cell = counts[(0, 1, 0)]
    fvseof_d2cell = counts[(1, 0, 0)]

    all_three = counts[(0, 0, 0)]

    colors = ['#f09b9b', '#fee090', '#74add1']
    venn = venn3_unweighted(subsets=(only_fseof, only_fvseof, fseof_fvseof, only_d2cell, fseof_d2cell, fvseof_d2cell, all_three),
          set_labels=('FSEOF', 'FVSEOF', 'D2Cell-pred'), set_colors=colors)

    for patch in venn.patches:
        patch.set_edgecolor('black')
        patch.set_alpha(1)
        patch.set_linewidth(0.5)

    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    venn3_plot(csv_file='../../Result/D2Cell-pred Result/Ecoli/ecoli_test_unseen_product_result.csv')
    venn3_plot(csv_file='../../Result/D2Cell-pred Result/Ecoli/ecoli_laser_dataset_result.csv', output='../../Result/fig4_e.pdf')