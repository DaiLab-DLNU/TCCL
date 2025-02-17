import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool as gep


class Graph_drug(nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, num_features_xc=92, num_features_pro=33,
                 num_features_ppi=1442):
        super(Graph_drug, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        # GCN encoder used for extracting drug features.
        self.molGconv1 = SAGEConv(num_features_xd, num_features_xd * 2, aggr='sum')
        self.molGconv2 = SAGEConv(num_features_xd * 2, num_features_xd * 4, aggr='sum')
        self.molGconv3 = SAGEConv(num_features_xd * 4, output_dim, aggr='sum')
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, 128)

        # GCN encoder used for extracting drug clique features.
        self.cliqueGconv1 = SAGEConv(num_features_xc, num_features_xc * 2, aggr='sum')
        self.cliqueGconv2 = SAGEConv(num_features_xc * 2, num_features_xc * 4, aggr='sum')
        self.cliqueGconv3 = SAGEConv(num_features_xc * 4, output_dim, aggr='sum')
        self.cliqueFC1 = nn.Linear(output_dim, 1024)
        self.cliqueFC2 = nn.Linear(1024, 128)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, mol_data, clique_data):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        xc, xc_edge_index, xc_batch = clique_data.x, clique_data.edge_index, clique_data.batch

        # Extracting drug features
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)  # global mean pooling
        # x = self.dropout2(self.relu(self.molFC1(x)))
        # x = self.dropout2(self.molFC2(x))

        # Extracting drug features
        xc = self.relu(self.cliqueGconv1(xc, xc_edge_index))
        xc = self.relu(self.cliqueGconv2(xc, xc_edge_index))
        xc = self.relu(self.cliqueGconv3(xc, xc_edge_index))
        xc = gep(xc, xc_batch)
        # xc = self.dropout2(self.relu(self.cliqueFC1(xc)))
        # xc = self.dropout2(self.cliqueFC2(xc))

        xcat = torch.cat((x, xc), dim=-1)

        return xcat


class Graph_protein(nn.Module): 
    def __init__(self, n_output=1, output_dim=128, num_features_pro=54):
        super(Graph_protein, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output

        # GCN encoder used for extracting protein features.
        self.proGconv1 = SAGEConv(num_features_pro, output_dim, aggr='sum')
        self.proGconv2 = SAGEConv(output_dim, output_dim*2, aggr='sum')
        self.proGconv3 = SAGEConv(output_dim*2, output_dim*2, aggr='sum')
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, 256)


        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, pro_graph):
        # seq_num = pro_graph.graph_num
        p_x, p_edge_index, p_batch = pro_graph.x, pro_graph.edge_index, pro_graph.batch


        # Extracting protein structural features from protein graphs.
        p_x = self.relu(self.proGconv1(p_x, p_edge_index))
        # p_x = torch.cat((torch.add(p_x, ppi_x), torch.sub(p_x, ppi_x)), -1)  # feature combination
        p_x = self.relu(self.proGconv2(p_x, p_edge_index))
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))
        p_x = gep(p_x, p_batch)
        # print("p_x.shape:", p_x.size())
        # p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        # p_x = self.dropout2(self.proFC2(p_x))

        return p_x
