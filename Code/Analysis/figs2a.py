import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bar_plot(publish_list):
    plt.figure(figsize=(7, 3), dpi=400)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['pdf.fonttype'] = 42
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    ind = np.arange(2000, 2024)
    width = 0.5

    data1 = publish_list[0]
    data2 = publish_list[1]
    data3 = publish_list[2]
    data4 = publish_list[3]
    data5 = publish_list[4]
    data6 = publish_list[5]
    data7 = publish_list[6]

    # 画柱状图
    p1 = plt.bar(ind, data1, width)
    p2 = plt.bar(ind, data2, width, bottom=data1)
    p3 = plt.bar(ind, data3, width, bottom=np.array(data1) + np.array(data2))
    p4 = plt.bar(ind, data4, width, bottom=np.array(data1) + np.array(data2) + np.array(data3))
    p5 = plt.bar(ind, data5, width, bottom=np.array(data1) + np.array(data2) + np.array(data3) + np.array(data4))
    p6 = plt.bar(ind, data6, width, bottom=np.array(data1) + np.array(data2) + np.array(data3) + np.array(data4) + np.array(data5))
    p7 = plt.bar(ind, data7, width,
                 bottom=np.array(data1) + np.array(data2) + np.array(data3) + np.array(data4) + np.array(data5) + np.array(data6))

    plt.xticks(ind, ('2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
                     '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'))
    plt.xticks(rotation=45, fontsize=7, ha='right')
    plt.xlabel('Year', fontsize=7)
    plt.ylabel('Literature counts')
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0]), ['Elsevier', 'Springer Nature', 'Oxford Academic', 'American Chemical Society', 'Wiley', 'MDPI', 'Other'])
    plt.savefig('../../Result/figS2_a.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    path = '../../Result/Database result/number_of_publisher_by_years.csv'
    df = pd.read_csv(path)
    publish_list = []
    pulisher_name = ['Elsevier', 'Springer Nature', 'Oxford Academic', 'American Chemical Society', 'Wiley', 'MDPI', 'Other']
    for pulisher in pulisher_name:
        df_publisher = df[df['publisher'] == pulisher]
        publish_list.append(df_publisher['number of article'].tolist())
    bar_plot(publish_list)