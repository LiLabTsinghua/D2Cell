import matplotlib.pyplot as plt
from matplotlib import rc

with open('../../Result/D2Cell-pred Result/Ecoli/train_loss.txt', 'r') as infile:
    lines = infile.readlines()

epoch_train = list()
loss_train = list()
epoch = 0
for line in lines:
    data = line.strip().split('\t')
    loss_line = float(data[0])
    epoch_train.append(epoch)
    epoch += 1
    loss_train.append(loss_line)
plt.figure(figsize=(1.5, 1.5))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 7})
plt.rcParams['pdf.fonttype'] = 42
plt.axes([0.12,0.12,0.83,0.83])

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#
plt.tick_params(direction='in')
plt.tick_params(which='major',length=1.5)
plt.tick_params(which='major',width=0.4)
#
plt.plot(epoch_train,loss_train, color='#159090', linestyle='dashed', linewidth=0.75, marker='s', markerfacecolor='#159090', markersize=1, label='Training')
plt.xlabel('Epoch', fontsize=7)
plt.ylabel('Cross Entropy Loss', fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
#
plt.savefig('../../Result/figs4_a.pdf', dpi=400, bbox_inches='tight')
plt.show()