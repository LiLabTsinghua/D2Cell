import random
import torch
import pandas as pd
from model import D2Cell_Model
import argparse
from dataset import D2CellDataset


def predict(train_loader, config, edge_index, edge_weight, model_path, product2idx_path, output_path='../predict_file/predict_yeast.xlsx'):
    model = D2Cell_Model(config, edge_index, edge_weight)
    model.to('cuda')
    model.load_state_dict(torch.load(model_path))
    df_product2idx = pd.read_excel(product2idx_path)
    model.eval()
    gt_list = []
    correct = 0
    total_num = 0
    predict_list = []
    product_list = []
    accuracy_list = []
    pert_list = []
    product_idx_list = []
    product_name_list = []
    for step, batch in enumerate(train_loader):
        batch = batch.to('cuda')
        output = model(batch)
        _, predicted = torch.max(output.data, 1)
        predict_list.extend(predicted.detach().cpu())
        gt_list.extend(batch.y.cpu())
        product_name_list.extend(batch.product_name)
        product_idx_list.extend(batch.product_idx.cpu())
        pert_list.extend(batch.pert_index)
        product_list.extend(batch.product_idx)
        correct += (predicted == batch.y).sum().item()
        total_num += len(batch.y)
    predict_list = [x.item() for x in predict_list]
    gt_list = [x.item() for x in gt_list]
    product_list = [x.item() for x in product_list]
    product_name_list = [x for x in product_name_list]
    # predict_list
    index_list = sorted(list(set(product_list)))
    all_TN = 0
    all_TP = 0
    all_FN = 0
    all_FP = 0
    for i in index_list:
        product_total = 0
        predict_true = 0
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for j in range(len(predict_list)):
            if product_list[j] == i:
                product_total += 1
                if predict_list[j] == gt_list[j]:
                    predict_true += 1
                    if predict_list[j] == 0:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if predict_list[j] == 0:
                        FP += 1
                    else:
                        FN += 1
        print('product', i, 'accuracy:', predict_true / product_total)
        accuracy_list.append(predict_true / product_total)
        all_TP += TP
        all_TN += TN
        all_FN += FN
        all_FP += FP
        print(TP, TN, FP, FN)
    print('correct', correct / total_num)
    print(all_TP, all_TN, all_FP, all_FN)
    print(accuracy_list)
    predict_data={'product index': product_list, 'pert index': pert_list, 'true label': gt_list, 'predict label': predict_list,
                  'product': product_name_list, 'inf_label_01': gt_list}
    predict_data = pd.DataFrame(predict_data)
    predict_data.to_excel(output_path, index=False)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of hidden layers')
    parser.add_argument('--num_gnn_layers', type=int, default=1, help='numbers of gnn layers.')
    parser.add_argument('--device', type=str, default='cuda', help='used device.')
    parser.add_argument('--num_met', type=int, default=2000, help='number of metabolites.')
    parser.add_argument('--output_path', type=str, default='../predict_file/ecoli_predict.xlsx',
                        help='result output path')
    parser.add_argument('--model_path', type=str, default='',
                        help='model_path of model.')
    parser.add_argument('--product2idx_path', type=str, default='../../Data/D2Cell-pred Data/Ecoli/ecoli-product2idx.xlsx',
                        help='ecoli-product2idx.xlsx')
    parser.add_argument('--test_dataset', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/ecoli_test_dataset.xlsx',
                        help='test dataset')
    parser.add_argument('--GEMs', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/iML1515_S.txt',
                        help='GEMs path')
    parser.add_argument('--IgnoreMets', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/IgnoreMets_iML1515.xlsx',
                        help='IgnoreMets path')
    args = parser.parse_args()


    output_path = args.output_path
    data = D2CellDataset(args.test_dataset, args.GEMs, args.IgnoreMets)
    dataloader, edge_index, edge_weight = data.get_dataloader()
    config = {'hidden_size': args.hidden_size,
              'num_gnn_layers': args.num_gnn_layers,
              'device': args.device,
              'num_met': args.num_met
              }
    test_loader = dataloader['test_loader']
    predict(test_loader, config, edge_index, edge_weight,
            model_path=args.model_path,
            output_path=output_path,
            product2idx_path=args.product2idx_path)
