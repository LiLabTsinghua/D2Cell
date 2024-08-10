import matplotlib.pyplot as plt
import pandas as pd


def draw_pie_figure(number_of_class, class_name):
    # Plot configuration
    plt.figure(figsize=(2, 2), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    # Plotting data
    explode0 = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    plt.pie(number_of_class, colors=['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6'], explode=explode0, radius=0.9,
            textprops={'fontsize': 6})
    # Add legend
    plt.legend(class_name, bbox_to_anchor=(2.2, 0.05), loc='lower right', fontsize=7, frameon=False)
    plt.savefig('../../Result/fig3_b.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    df_path = '../../Result/Database result/number_of_products_study.csv'
    df = pd.read_csv(df_path)
    number_of_class = df['counts'].tolist()
    class_name = df['product class'].tolist()
    draw_pie_figure(number_of_class, class_name)