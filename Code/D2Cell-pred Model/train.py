import argparse
import os
import matplotlib.pyplot as plt
import torch
from model import D2Cell_Model
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from collections import deque
from dataset import D2CellDataset


def train(train_loader, config, edge_index, edge_weight, save_path='../save_model/ecoli_model/ecoli_GEMs_'):
    model = D2Cell_Model(config, edge_index, edge_weight)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    loss_fn = CrossEntropyLoss()
    loss_fn.to('cuda')
    losses = []
    best_loss = 100
    epochs = args.epochs
    print('Start Training...')
    saved_files = deque(maxlen=20)
    for epoch in tqdm(range(epochs), desc="Processing"):
        model.train()
        train_loss = 0
        step_number = 0
        for step, batch in enumerate(train_loader):
            batch = batch.to('cuda')
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()
            mean_loss = torch.mean(loss)
            train_loss += mean_loss.item()
            step_number = step+1
        train_loss = train_loss/step_number
        losses.append(train_loss)
        print('-------------------------')
        print('Epoch {} train loss: {:.8f}'.format(epoch + 1, train_loss))

        if train_loss < best_loss:
            best_loss = train_loss
            print('Best loss: {:.8f}\t'.format(best_loss), 'Saving model...\t', 'best epoch:', epoch+1)
            torch.save(model.state_dict(), save_path+str(epoch)+'.pth')
            model_file = save_path+str(epoch)+'.pth'
            saved_files.append(model_file)
            if len(saved_files) == 5:
                oldest_file = saved_files.popleft()
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)

    print(losses)
    plt.plot(range(0, epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of hidden layers')
    parser.add_argument('--num_gnn_layers', type=int, default=1,help='numbers of gnn layers.')
    parser.add_argument('--device', type=str, default='cuda', help='used device.')
    parser.add_argument('--num_met', type=int, default=2000, help='number of metabolites.')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs.')
    parser.add_argument('--train_dataset', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/ecoli_train_dataset.xlsx',
                        help='train dataset')
    parser.add_argument('--GEMs', type=str, default='../../Data/D2Cell-pred Data/Ecoli/iML1515_S.txt',
                        help='GEMs path')
    parser.add_argument('--IgnoreMets', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/IgnoreMets_iML1515.xlsx',
                        help='IgnoreMets path')
    parser.add_argument('--save_model', type=str, default='',
                        help='path for saving model.')
    args = parser.parse_args()

    data = D2CellDataset(data_file=args.train_dataset, graph_file=args.GEMs,
                        delete_meta_path=args.IgnoreMets)
    dataloader, edge_index, edge_weight = data.get_dataloader()
    config = {'hidden_size': args.hidden_size,
              'num_gnn_layers': args.num_gnn_layers,
              'device': args.device,
              'num_met': args.num_met
              }
    train_loader = dataloader['train_loader']
    train(train_loader, config, edge_index, edge_weight, save_path=args.save_model)

