import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../Result/D2Cell-pred Result/Ecoli/test_valid_acc.csv')
plt.figure(figsize=(1.5, 1.5))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 7})
plt.axes([0.12,0.12,0.83,0.83])
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#
plt.tick_params(direction='in')
plt.tick_params(which='major',length=1.5)
plt.tick_params(which='major',width=0.4)
colors = ['#fd8d3c','#74a9cf','#2166ac','#b2182b','#159090']
for i in range(5):
    test_acc = df['test'+str(i+1)+'_acc'].tolist()
    test_acc = [float(acc) for acc in test_acc]
    epochs = df['epoch'].tolist()
    epochs = [float(epoch) for epoch in epochs]
    plt.plot(epochs, test_acc, color=colors[i], linestyle='dashed', linewidth=0.75,marker='o',markerfacecolor='#159090', markersize=0.5,label='Test '+str(i+1))
plt.xlabel('Epoch', fontsize=7)
plt.ylabel('Accuracy on Test Set', fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(frameon=False, prop={"size":7})
#
ax = plt.gca()
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
#
plt.savefig('../../Result/figs4_c.pdf', dpi=400, bbox_inches='tight')
plt.show()