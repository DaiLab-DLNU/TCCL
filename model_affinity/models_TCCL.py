import torch
import torch.nn as nn

from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj
from TCCL_graph import *
from TCCL_sequence import *
from utils_data import *

class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings




class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings


class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_a + (1 - self.lam) * lori_b, torch.cat((za_proj, zb_proj), 1)


class TCCLDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, dropout_rate, embedding_dim=128):
        super(TCCLDTA, self).__init__()

        self.output_dim = embedding_dim * 2
        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        self.drug_sequence_conv = Sequence_drug()
        self.target_sequence_conv = Sequence_protein()
        self.drug_graph_conv = Graph_drug()
        self.target_graph_conv = Graph_protein()

        self.drug_seq_contrast = Contrast(256, embedding_dim, tau, lam)
        self.target_seq_contrast = Contrast(256, embedding_dim, tau, lam)
        self.drug_graph_contrast = Contrast(256, embedding_dim, tau, lam)
        self.target_graph_contrast = Contrast(256, embedding_dim, tau, lam)

    def forward(self, affinity_graph, drug_seq_batchs, target_seq_batchs, drug_graph_batchs, clique_seq_batchs,
                target_graph_batchs, drug_pos, target_pos):
        num_d = affinity_graph.num_drug

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        for batch_idx, data in enumerate(drug_seq_batchs):
            # print(data)
            drug_seq_batchs = torch.tensor(data)
        drug_seq_embedding = self.drug_sequence_conv(drug_seq_batchs)
        # print('drug_seq_embedding',drug_seq_embedding.size())
        for batch_idx, data in enumerate(target_seq_batchs):
            target_seq_batchs = torch.Tensor(data)
        target_seq_embedding = self.target_sequence_conv(target_seq_batchs)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs[0], clique_seq_batchs[0])
        target_graph_embedding = self.target_graph_conv(target_graph_batchs[0])
        drug_pos = drug_pos.to('cuda:0')
        target_pos = target_pos.to('cuda:0')
        dru_loss1, drug_embedding1 = self.drug_graph_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding,
                                                              drug_pos)
        tar_loss1, target_embedding1 = self.target_graph_contrast(affinity_graph_embedding[num_d:],
                                                                  target_graph_embedding,
                                                                  target_pos)
        dru_loss2, drug_embedding2 = self.drug_seq_contrast(affinity_graph_embedding[:num_d], drug_seq_embedding,
                                                            drug_pos)
        tar_loss2, target_embedding2 = self.target_seq_contrast(affinity_graph_embedding[num_d:], target_seq_embedding,
                                                                target_pos)
        a = torch.add(drug_embedding1, drug_embedding2)
        b = torch.add(target_embedding1, target_embedding2)

        return dru_loss1 + tar_loss1 + dru_loss2 + tar_loss2, a, b



class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y
        # print(drug_id,target_id)

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings
