import pandas as pd
import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx


class D2CellDataset:
    def __init__(self, data_file, graph_file, delete_meta_path):
        """
        Initialize the dataset
        ----------
        :param edges: list
            A list of edges, where each element is a tuple (u, v) representing an undirected edge.
        :param node_features: dict
            A dictionary where the keys are nodes and the values are feature vectors associated with the nodes.
        """
        self.data_path = data_file
        self.graph_path = graph_file
        self.delete_meta_path = delete_meta_path

    def extract_matrix(self, file_path):
        '''
        Extracted Metabolite-Reaction Relationship Matrix
        ----------
        :param file_path: str
            reaction and metabolite file path
        :returns: relationship matrix, row labels (metabolite names), column labels (reaction names)
        '''
        matrix_data = []
        met_labels = []
        rxn_labels = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Extract column label (reaction name)
            rxn_labels = lines[0].strip().split()[1:]
            # Extract line labels (metabolite names) as well as numerical data
            for line in lines[1:]:
                elements = line.strip().split()
                met_labels.append(elements[0])
                row_data = [float(val) for val in elements[1:]]
                matrix_data.append(row_data)
        matrix = np.array(matrix_data)
        return matrix, met_labels, rxn_labels

    def find_index(self, input_row):
        '''
        Extracts the indexes of the positive (reaction input) and negative (reaction output) numbers in the input row,
        ----------
        : param input_row: Row of the input relational matrix
        '''
        positive_indices = [index for index, value in enumerate(input_row) if value > 0]
        negative_indices = [index for index, value in enumerate(input_row) if value < 0]
        return positive_indices, negative_indices

    def matrix2graph(self, matrix, delete_list):
        '''
        Converting a relationship matrix into a graph
        ----------
        :param matrix: np.ndarray
            Relationship matrix
        :param delete_list: list
            List of metabolite names to delete
        :returns: Edge indices and weights of the generated graph
        '''
        edge_list = [['source', 'target', 'importance']]
        matrix = np.delete(matrix, delete_list, axis=0)
        my_set = set()
        number = 0
        matrix = zip(*matrix)
        for row in matrix:
            positive_index, negative_index = self.find_index(row)
            number += 1
            if positive_index and negative_index:
                for source in negative_index:
                    for target in positive_index:
                        my_set.add((source, target, 1))
        my_list = [list(t) for t in my_set]
        edge_list.extend(my_list)
        edge_list = pd.DataFrame(edge_list[1:], columns=edge_list[0])
        G = nx.from_pandas_edgelist(edge_list, source='source',
                                    target='target', edge_attr=['importance'],
                                    create_using=nx.DiGraph())
        # print(G.number_of_nodes(), G.number_of_edges())
        edge_index_ = [(e[0], e[1]) for e in G.edges]
        edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        edge_attr = nx.get_edge_attributes(G, 'importance')
        importance = np.array([edge_attr[e] for e in G.edges])
        edge_weight = torch.Tensor(importance)
        return edge_index, edge_weight

    def get_dataloader(self):
        """
        Get dataloaders for training and testing
        ----------
        :returns: DataLoader for training and testing sets
        """
        matrix, row_labels, col_labels = self.extract_matrix(self.graph_path)
        delete_meta = pd.read_csv(self.delete_meta_path)['met_id'].tolist()
        delete_list = []
        for i in range(len(row_labels)):
            if row_labels[i] in delete_meta:
                delete_list.append(i)
        edge_index, edge_weight = self.matrix2graph(matrix, delete_list)
        df = pd.read_csv(self.data_path)
        flux_data = []
        for index, row in df.iterrows():
            if index >= 1:
                output = torch.tensor([row['inf_label_01']])
                overexpress_list = []
                knock_list = eval(row['pert index'])
                product_name = row['product']
                flux_data.append(Data(product_idx=torch.tensor(row['index in gem']),
                                      y=output, ov_idx=overexpress_list, ko_idx=knock_list, edge_index=edge_index,
                                      edge_weight=edge_weight, pert_index=knock_list, product_name=product_name))
        random.shuffle(flux_data)
        test_loader = DataLoader(flux_data,
                                 batch_size=16, shuffle=True)
        train_loader = DataLoader(flux_data,
                                  batch_size=16, shuffle=True)
        dataloader = {'train_loader': train_loader,
                      'test_loader': test_loader}
        return dataloader, edge_index, edge_weight