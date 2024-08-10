import matplotlib.pyplot as plt
import numpy as np
from IE_accuracy import calculate_accuracy


# Data for plotting
def bar_f1score(scores, fig_name):
    methods = ['Qwen1.5-110B', 'Gemini Pro', ' Claude-3', 'GPT-4', 'Qwen-Lora']
    shots = ['Precision', 'Recall', 'F1 Score']
    plt.rcParams['pdf.fonttype'] = 42
    # Plot details
    bar_width = 0.15  # width of the bars
    index = np.arange(len(shots))
    colors = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#41b6c4', '#225ea8']
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(2, 1.5), dpi=400)
    fig.subplots_adjust(left=0.2, right=1.5, top=0.8, bottom=0.2)
    for i, method in enumerate(methods):
        ax.bar(index + i * bar_width, scores[i], color=colors[i],
               width=bar_width, label=method, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(index + bar_width + bar_width + bar_width / 50)
    ax.set_xticklabels(shots)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=5)
    plt.ylabel('NER performance (%)', fontsize=7)
    # Set the linewidth of the plot spines
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.tick_params(axis='y', direction='in', width=0.5, which='both', length=1.5)
    plt.tick_params(axis='x', direction='in', which='both', width=0.5, length=1.5)
    ax.set_ylim([40, 105])

    # Adjust the layout to be tight within the small figure size
    plt.tight_layout()
    plt.savefig('../../Result/figs1_{}.pdf'.format(fig_name), dpi=400, bbox_inches='tight')
    # Show the plot
    plt.show()


if __name__ == '__main__':
    IE_score = calculate_accuracy()
    IE_score = [[value * 100 for value in sublist] for sublist in IE_score]
    bar_f1score(IE_score, fig_name='a')