import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches


process_1_data = [61, 0, 130.0 - 61]  # NER time
process_2_data = [89, 130.0 - 89]  # RE time
process_3_data = [10, 130.0 - 10]  # Python time

fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(aspect="equal"))
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
size = 0.2
colors = ['#c2c2f0','#ffb3e6', '#c2f0c2']
hole_color = 'white'
labels = ['Qwen lora', 'Qwen1.5-110B-Chat', 'Code Run Time']

ax.pie(process_1_data, radius=1, colors=[colors[0], colors[1], hole_color], startangle=90, wedgeprops=dict(width=size, edgecolor='w'))
ax.pie(process_2_data, radius=1-size, colors=[colors[1], hole_color], startangle=90, wedgeprops=dict(width=size, edgecolor='w'))
ax.pie(process_3_data, radius=1-size-size, colors=[colors[2], hole_color], startangle=90, wedgeprops=dict(width=size, edgecolor='w'))
legend_colors = [mpatches.Patch(color=colors[i]) for i in range(len(colors))]
ax.legend(loc='lower right', labels=labels,handles=legend_colors, bbox_to_anchor=(1.1, -0.1))
ax.set(aspect="equal")

labels = ['Code Run Time', 'Relation Extraction', 'Named Entity Recognition', 'Execution Time Per Paper (s)']

for i, label in enumerate(labels):
    angle = (process_1_data[0] if i == 0 else process_2_data[0] if i == 1 else process_3_data[0]) / 2 + 90
    x = 0
    if i == 0:
        y = (1 - size / 2) - 0.4
    if i == 1:
        y = (1 - size / 2) - 0.2
    if i == 2:
        y = (1 - size / 2)
    if i == 3:
        x = -0.5
        y = (1 - size / 2)-1
    ax.annotate(label, (x, y), fontsize=8, ha='left', va='center')

for i, value in enumerate([process_1_data[0], 21, process_2_data[0], process_3_data[0]]):
    if i == 0:
        x = (1 - size / 2)-1
        y = (1 - size / 2)
        ax.text(x, y + 0.1, f'{value}', fontsize=8, ha='center')
    if i == 1:
        x = (1 - size / 2) - 1.4
        y = (1 - size / 2) - 2
        ax.text(x, y + 0.1, f'{value}', fontsize=8, ha='center')
    if i == 2:
        x = (1 - size / 2) - 0.3
        y = (1 - size / 2) - 1.72
        ax.text(x, y + 0.1, f'{value}', fontsize=8, ha='center')
    if i == 3:
        x = (1 - size / 2) - 1.15
        y = (1 - size / 2) - 0.6
        ax.text(x, y + 0.1, f'{value}', fontsize=8, ha='center')

#
plt.savefig('../../Result/fig2_b.pdf', dpi=400, bbox_inches='tight')
plt.show()