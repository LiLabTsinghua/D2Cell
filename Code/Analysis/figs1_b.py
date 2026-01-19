import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar(ylabel,label_list,value_list,rotation=45, output=''):

    # Setting General Parameters
    plt.figure(figsize=(2, 1.65), dpi=400)
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.subplots_adjust(left=0.2, right=1.5, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    bar_width = 0.4
    index = np.arange(len(label_list))
    colors = ['#FFFFF0', '#ffffd9', '#edf8b1','#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']
    plt.bar(index, value_list, width=bar_width,
            color=colors, linewidth=0.5, edgecolor='black')
    plt.xticks(index, label_list, fontsize=7, rotation=rotation, ha='right')
    plt.ylim(0, 105)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=7)

    # Setting Axis Parameters
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    plt.ylabel(ylabel, fontsize=8)
    plt.legend(label_list, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=5)
    plt.tight_layout()
    plt.savefig(output, dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    xlabel = ['Qwen1.5-110B', 'Qwen1.5-14B', 'Qwen3-30B', 'Gemini Pro', ' Claude-3', 'GPT-4', 'Llama3', 'Llama3-Lora',
               'Qwen1.5-Lora', 'Qwen3-Lora']
    df = pd.read_csv('../../Result/NER Data/NER_execution_time.csv')
    value_list = df['Execution time(min)'].tolist()
    bar(ylabel='NER execution time (min)', label_list=xlabel, value_list=value_list, rotation=45, output='../../Result/figs1_b.pdf')