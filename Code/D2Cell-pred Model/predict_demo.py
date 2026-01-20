import random
import torch
import pandas as pd
from model import D2Cell_Model
import argparse
from dataset import D2CellDataset


def predict(train_loader, config, edge_index, edge_weight, model_path, product2idx_path, output_path='../predict_file/predict_yeast.xlsx'):
    """
        Loads a trained model, performs inference on the provided dataset,
        and saves the prediction results to a CSV file.
        """

    # --- Model Initialization ---
    # Initialize the model structure with the given configuration and graph data
    model = D2Cell_Model(config, edge_index, edge_weight)

    # Move the model to the specified device (GPU)
    model.to('cuda')

    # Load the trained model weights (state dictionary) from the specified file path
    model.load_state_dict(torch.load(model_path))

    # Load the mapping file for product indices (though strictly not used in the loop, likely for reference)
    df_product2idx = pd.read_csv(product2idx_path)

    # Set the model to evaluation mode (disables Dropout and BatchNorm updates)
    model.eval()

    # --- Prediction Loop ---
    # Initialize lists to store results
    predict_list = []  # Stores predicted labels
    product_list = []  # Stores product indices
    pert_list = []  # Stores perturbation indices
    product_idx_list = []  # Stores product indices (duplicate of product_list based on code logic)
    product_name_list = []  # Stores product names

    # Iterate over the data loader (batches)
    for step, batch in enumerate(train_loader):
        # Move the batch data to the GPU
        batch = batch.to('cuda')

        # Forward pass: Generate model outputs
        output = model(batch)

        # Get the predicted class by finding the index of the maximum value in the output
        _, predicted = torch.max(output.data, 1)

        # --- Result Collection ---
        # Detach from computation graph, move to CPU, and extend the results list
        predict_list.extend(predicted.detach().cpu())

        # Extend metadata lists from the batch
        product_name_list.extend(batch.product_name)
        product_idx_list.extend(batch.product_idx.cpu())
        pert_list.extend(batch.pert_index)
        product_list.extend(batch.product_idx)

    # --- Post-processing ---
    # Convert tensor items to standard Python scalars for saving
    predict_list = [x.item() for x in predict_list]
    product_list = [x.item() for x in product_list]
    product_name_list = [x for x in product_name_list]
    # predict_list

    predict_data={'product index': product_list, 'pert index': pert_list, 'predict label': predict_list,
                  'product': product_name_list}

    # Save the DataFrame to a CSV file without the index column
    predict_data = pd.DataFrame(predict_data)
    predict_data.to_csv(output_path, index=False)


if __name__ == '__main__':
    # Set random seeds for Python and PyTorch to ensure consistent results
    random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='size of hidden layers')
    parser.add_argument('--num_gnn_layers', type=int, default=1, help='numbers of gnn layers.')
    parser.add_argument('--device', type=str, default='cuda', help='used device.')
    parser.add_argument('--num_met', type=int, default=2000, help='number of metabolites.')
    parser.add_argument('--output_path', type=str, default='../../predict_file/ecoli_predict.csv',
                        help='result output path')
    parser.add_argument('--model_path', type=str, default='../../save_model/ecoli_model/ecoli_D2Cell_pred_model.pth',
                        help='model_path of model.')
    parser.add_argument('--product2idx_path', type=str, default='../../Data/D2Cell-pred Data/Ecoli/ecoli-product2idx.csv',
                        help='ecoli-product2idx.xlsx')
    parser.add_argument('--test_dataset', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/ecoli_test_dataset.csv',
                        help='test dataset')
    parser.add_argument('--GEMs', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/iML1515_S.txt',
                        help='GEMs path')
    parser.add_argument('--IgnoreMets', type=str,
                        default='../../Data/D2Cell-pred Data/Ecoli/IgnoreMets_iML1515.csv',
                        help='IgnoreMets path')
    # Parse the arguments
    args = parser.parse_args()

    # --- Data Preparation ---
    output_path = args.output_path

    # Initialize the custom dataset class with the provided paths
    data = D2CellDataset(args.test_dataset, args.GEMs, args.IgnoreMets)

    # specific method to get dataloaders and graph structure (edge_index, edge_weight)
    dataloader, edge_index, edge_weight = data.get_dataloader()

    # Create configuration dictionary from arguments
    config = {
        'hidden_size': args.hidden_size,
        'num_gnn_layers': args.num_gnn_layers,
        'device': args.device,
        'num_met': args.num_met
    }

    # Extract the test data loader from the dataloader dictionary
    test_loader = dataloader['test_loader']

    # Run the prediction function
    predict(test_loader, config, edge_index, edge_weight,
            model_path=args.model_path,
            output_path=output_path,
            product2idx_path=args.product2idx_path)
