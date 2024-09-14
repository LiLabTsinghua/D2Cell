import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv, GCNConv


class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FlattenMLP(nn.Module):
    def __init__(self, reaction_size, hidden_size):
        """
        Faltten Multi-layer perceptron
        :param reaction_size: sizes of the reaction
        :param hidden_size: sizes of the hidden layers
        """
        super(FlattenMLP, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(reaction_size * hidden_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        """
        DecoderBlock
        :param embed_size: The embedding size of each input sequence element
        :param heads: The number of parallel attention heads
        :param forward_expansion: The expansion factor, mapping the embedding dimension to a wider dimension
        :param dropout: Regularization parameter
        """
        super(DecoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention, attention_weight = self.attention(x, x, x)
        x = self.norm1(attention + x)
        x = self.dropout(x)

        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        out = self.dropout(out)
        return out


class D2Cell_Model(torch.nn.Module):
    """
    D2Cell model

    """

    def __init__(self, args, D2Cell_edge_index, D2Cell_edge_weight):
        """
        :param args: arguments dictionary
        """

        super(D2Cell_Model, self).__init__()
        self.args = args
        self.num_pert = 3100
        hidden_size = args['hidden_size']
        self.num_layers = args['num_gnn_layers']
        self.num_met = args['num_met']
        self.num_product = 20
        self.D2Cell_edge_index = D2Cell_edge_index.to(self.args['device'])
        self.D2Cell_edge_weight = D2Cell_edge_weight.to(self.args['device'])
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene/product embedding dictionary lookup
        self.pert_emb = nn.Embedding(self.num_pert, hidden_size, max_norm=True)
        self.product_emb = nn.Embedding(self.num_product, hidden_size, max_norm=True)
        self.meta_graph_emb = nn.Embedding(self.num_met, hidden_size, max_norm=True)

        # transformation layer
        self.emb_mlp = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.product_mlp = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.down_grade_emb = FlattenMLP(self.num_met, hidden_size)

        self.layers_gem_gnn = torch.nn.ModuleList()
        # use different GNN for GEM embedding
        for i in range(1, self.num_layers + 1):
            self.layers_gem_gnn.append(SGConv(hidden_size, hidden_size, 2))

        self.layers_meta_gnn = torch.nn.ModuleList()
        # use different GNN for meta embedding
        for i in range(1, self.num_layers + 1):
            self.layers_meta_gnn.append(SGConv(hidden_size, hidden_size, 2))

        # decoder shared MLP
        self.pert_mlp = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='linear')

        self.transformer_decoder = DecoderBlock(embed_size=384, heads=4, forward_expansion=2, dropout=0.05)

        self.ff_layer = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.fc_out = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        """
        Forward pass of the model
        """
        product_index, pert_index = data.product_idx, data.pert_index
        edge_index = data.edge_index.to(self.args['device'])
        edge_weight = data.edge_weight.to(self.args['device'])
        unique_batch = len(data.batch.unique())
        product_add_list = list(range(unique_batch))

        pos_emb = self.meta_graph_emb(
            torch.LongTensor(list(range(self.num_met))).repeat(unique_batch, ).to(self.args['device']))
        for idx, layer in enumerate(self.layers_gem_gnn):
            pos_emb = layer(pos_emb, edge_index, edge_weight)
            if idx < len(self.layers_gem_gnn) - 1:
                pos_emb = pos_emb.relu()

        base_emb = pos_emb
        base_emb = self.emb_mlp(base_emb)
        base_emb = base_emb.reshape(unique_batch, self.num_met, -1)
        ## get perturbation index and embeddings

        meta_emb = self.meta_graph_emb(torch.LongTensor(list(range(self.num_met))).to(self.args['device']))
        for idx, layer in enumerate(self.layers_meta_gnn):
            meta_emb = layer(meta_emb, self.D2Cell_edge_index, self.D2Cell_edge_weight)
            if idx < len(self.layers_meta_gnn) - 1:
                meta_emb = meta_emb.relu()
        product_emb = meta_emb[product_index]
        product_emb = self.product_mlp(product_emb)

        pert_emb_all = self.pert_emb(torch.arange(self.num_pert, device=self.args['device']))
        pert_emb_sum_list = []
        for i in range(unique_batch):
            # Select the perturbation embeddings needed for the current graph from pert_emb_all and accumulate them
            pert_emb_sum = pert_emb_all[pert_index[i]].sum(dim=0)
            pert_emb_sum_list.append(pert_emb_sum)

        pert_emb = torch.stack(pert_emb_sum_list, dim=0)
        pert_emb = self.pert_mlp(pert_emb)
        base_emb = self.down_grade_emb(base_emb)
        base_emb = torch.cat((base_emb, pert_emb), dim=1)
        base_emb = torch.cat((base_emb, product_emb), dim=1)

        base_emb = self.ff_layer(base_emb)
        base_emb = base_emb.view(unique_batch, -1)
        output = self.fc_out(base_emb)
        output = self.softmax(output)
        return output


        
