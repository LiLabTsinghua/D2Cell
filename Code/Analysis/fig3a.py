import matplotlib.pyplot as plt
import pandas as pd


def draw_pie_figure(strain_list):
    # Plot configuration
    plt.figure(figsize=(2, 2), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    # Plotting data
    explode0 = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
    plt.pie(strain_list, colors=['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6'], explode=explode0, radius=0.9,
            textprops={'fontsize': 7})
    legend = plt.legend(['E. coli', 'S. cerevisiae', 'C. glutamicum', 'Y. lipolytica', 'B. subtilis', 'P. pastoris', 'P. putida', 'B. licheniformis', 'Others'],
               bbox_to_anchor=(1.65, 0.05), loc='lower right', fontsize=7, frameon=False)
    for text in legend.get_texts()[:8]:
        text.set_fontstyle('italic')
    plt.savefig('../../Result/fig3_a.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    df_path = '../../Result/Database result/number_of_organisms_study.csv'
    df = pd.read_csv(df_path)
    strain_number_list = df['counts'].tolist()
    strain_number_list = sorted(strain_number_list, reverse=True)
    all_data = sum(strain_number_list)
    data_list = []
    for i in range(7):
        data_list.append(strain_number_list[i])
    data_list.append(strain_number_list[9])
    data_list.append(all_data-sum(data_list))
    strain_number_list = data_list
    print(len(strain_number_list))
    draw_pie_figure(strain_number_list)