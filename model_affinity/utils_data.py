import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from lifelines.utils import concordance_index

from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from torch_geometric.data import Batch, InMemoryDataset
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric import data as DATA
import torch
import pickle
from torch_geometric.loader import DataLoader
import warnings

warnings.filterwarnings("ignore")

class DTADataset(Dataset):
    def __init__(self,smile_list, seq_list, label_list,mol_data = None,clique_data=None, ppi_index = None):
        super(DTADataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.clique_graph = clique_data
        self.ppi_index = ppi_index

    def len(self):
        return len(self.smile_list)

    def get(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels =self.label_list[index]

        drug_size, drug_features, drug_edge_index = self.smile_graph[smile]
        clique_size, clique_features, clique_edge_index = self.clique_graph[smile]

        seq_size = len(seq)
        seq_index =self.ppi_index[seq]

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        GCNData_smile = DATA.Data(x=torch.Tensor(drug_features), edge_index=torch.LongTensor(drug_edge_index).transpose(1, 0), y=torch.FloatTensor([labels]))
        GCNData_smile.__setitem__('c_size', torch.LongTensor([drug_size]))
        GCNData_clique = DATA.Data(x=torch.Tensor(clique_features), edge_index=torch.LongTensor(clique_edge_index).transpose(1, 0), y=torch.FloatTensor([labels]))
        GCNData_clique.__setitem__('c_size', torch.LongTensor([clique_size]))
        GCNData_seq = DATA.Data(y=torch.FloatTensor([labels]),seq_num =torch.LongTensor([seq_index])) # The seq_index indicates the node number of the protein in the PPI graph.
        print('pro_data.seq_num', GCNData_seq.seq_num)
        GCNData_seq.__setitem__('c_size', torch.LongTensor([seq_size]))
        return GCNData_smile, GCNData_seq, GCNData_clique
class DTADataset1(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset1, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]), y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset):
    affinity = pickle.load(open('/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/' + dataset + '/affinities', 'rb'),
                           encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj

def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class proGraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', graph=None, index=None, type=None):
        super(proGraphDataset, self).__init__(root)
        self.type = type
        self.index = index
        self.process(graph, index)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graph, index):
        data_list = []
        count = 0
        for key in index:
            size, features, edge_index = graph[key]
            # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index),
                                graph_num=torch.LongTensor([count]))
            GCNData.__setitem__('c_size', torch.LongTensor([size]))
            count += 1
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DTADataset2(Dataset):
    def __init__(self, smile_list, seq_list, label_list, mol_data=None, clique_data=None, ppi_index=None):
        super(DTADataset2, self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.clique_graph = clique_data
        self.ppi_index = ppi_index

    def len(self):
        return len(self.smile_list)

    def get(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels = self.label_list[index]

        compoundint = torch.from_numpy(label_smiles(smile, CHARISOSMISET, 100))
        proteinint = torch.from_numpy(label_sequence(seq, CHARPROTSET, 1200))
        labelsint = np.float(labels)

        drug_size, drug_features, drug_edge_index = self.smile_graph[smile]
        clique_size, clique_features, clique_edge_index = self.clique_graph[smile]

        seq_size = len(seq)
        seq_index = self.ppi_index[seq]

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        GCNData_smile = DATA.Data(x=torch.Tensor(drug_features),
                                  edge_index=torch.LongTensor(drug_edge_index).transpose(1, 0),
                                  y=torch.FloatTensor([labels]))
        GCNData_smile.__setitem__('c_size', torch.LongTensor([drug_size]))
        GCNData_clique = DATA.Data(x=torch.Tensor(clique_features),
                                   edge_index=torch.LongTensor(clique_edge_index).transpose(1, 0),
                                   y=torch.FloatTensor([labels]))
        GCNData_clique.__setitem__('c_size', torch.LongTensor([clique_size]))
        GCNData_seq = DATA.Data(y=torch.FloatTensor([labels]), seq_num=torch.LongTensor(
            [seq_index]))  # The seq_index indicates the node number of the protein in the PPI graph.
        GCNData_seq.__setitem__('c_size', torch.LongTensor([seq_size]))

        return GCNData_smile, GCNData_seq, GCNData_clique, compoundint, proteinint, labelsint

def collate_fn1(batch_data, max_d=100, max_p=1200):

    N = len(batch_data)
    compound_new = torch.zeros((N, max_d), dtype=torch.long)
    protein_new = torch.zeros((N, max_p), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.float)
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        compoundstr, proteinstr, label = pair[-3], pair[-2], pair[-1]
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, max_d))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, max_p))
        protein_new[i] = proteinint
        labels_new[i] = np.float(label)
    return (compound_new, protein_new, labels_new)






def collate_fn(batch_data):
    print('collate_fn is running!')
    # N = len(batch_data)
    # compound_new = torch.zeros((N, max_d), dtype=torch.long)
    # protein_new = torch.zeros((N, max_p), dtype=torch.long)
    # labels_new = torch.zeros(N, dtype=torch.float)
    # GCNData_smile_list = []
    # GCNData_seq_list = []
    # GCNData_clique_list = []

    GCNData_smile_list = [sample[0] for sample in batch_data]
    GCNData_seq_list = [sample[1] for sample in batch_data]
    GCNData_clique_list = [sample[2] for sample in batch_data]
    compound_new = [sample[3] for sample in batch_data]
    protein_new = [sample[4] for sample in batch_data]
    labels_new = [sample[5] for sample in batch_data]

    # for i, pair in enumerate(batch_data):
    #     pair = pair.strip().split()
    #     GCNData_smile, GCNData_seq, GCNData_clique, compoundstr, proteinstr, label = pair[-6], pair[-5], pair[-4],pair[-3], pair[-2], pair[-1]
    #     GCNData_smile_list[i] = GCNData_smile
    #     GCNData_seq_list[i] = GCNData_seq
    #     GCNData_clique_list[i] = GCNData_clique
    #     compoundstr, proteinstr, label = pair[-3], pair[-2], pair[-1]
    #     compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, max_d))
    #     compound_new[i] = compoundint
    #     proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, max_p))
    #     protein_new[i] = proteinint
    #     labels_new[i] = np.float(label)
    return (GCNData_smile_list, GCNData_seq_list, GCNData_clique_list, compound_new, protein_new, labels_new)


def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


class GraphDatasetPro(InMemoryDataset):
    def __init__(self, root='/tmp', graph=None, index=None, type=None):
        super(GraphDatasetPro, self).__init__(root)
        self.type = type
        self.index = index
        self.process(graph, index)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graph, index):
        data_list = []
        count = 0
        for key in index:
            size, features, edge_index = graph[key]
            # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index),
                                graph_num=torch.LongTensor([count]))
            GCNData.__setitem__('c_size', torch.LongTensor([size]))
            count += 1
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def proGraph(graph_data, index, device):
    proGraph_dataset = GraphDatasetPro(graph=graph_data, index=index, type='pro')
    proGraph_loader = DataLoader(proGraph_dataset, batch_size=len(graph_data), shuffle=False)
    pro_graph = None
    for batchid, batch in enumerate(proGraph_loader):
        pro_graph = batch.x.to(device), batch.edge_index.to(device), batch.graph_num.to(device), batch.batch.to(device)
    return pro_graph


# class CustomDataSet(Dataset):
#     def __init__(self, pairs):
#         self.pairs = pairs
#         self._indices = None  # 初始化_indices属性
#
#     def get(self, item):
#         return self.pairs[item]
#
#     def len(self):
#         return len(self.pairs)

class CustomDataSet(Dataset):
    def __init__(self, drug_seq_embeddings, transform=None):
        super(CustomDataSet, self).__init__()
        self.drug_seq_embeddings = drug_seq_embeddings
        self.transform = transform

    def len(self):
        return len(self.drug_seq_embeddings)

    def get(self, idx):
        drug_seq_embedding = self.drug_seq_embeddings[idx]

        if self.transform:
            drug_seq_embedding = self.transform(drug_seq_embedding)

        return drug_seq_embedding


def positive_create(dataset, dataset_path, num_pos, pos_threshold):
    affinity_mat = load_data(dataset)

    train_file = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_index = []
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'S1_test_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]

    train_dataset = DTADataset1(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset1(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity_mat, num_pos, pos_threshold)

    return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos


def get_affinity_graph(dataset, adj, num_pos, pos_threshold):
    dataset_path = '../data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]

    dt_ = adj.copy()
    dt_ = np.where(dt_ >= pos_threshold, 1.0, 0.0)
    dtd = np.matmul(dt_, dt_.T)
    dtd = dtd / dtd.sum(axis=-1).reshape(-1, 1)
    dtd = np.nan_to_num(dtd)
    dtd += np.eye(num_drug, num_drug)
    dtd = dtd.astype("float32")
    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    dAll = dtd + d_d
    drug_pos = np.zeros((num_drug, num_drug))
    for i in range(len(dAll)):
        one = dAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-dAll[i, one])
            sele = one[oo[:num_pos]]
            drug_pos[i, sele] = 1
        else:
            drug_pos[i, one] = 1
    drug_pos = sp.coo_matrix(drug_pos)
    drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)

    td_ = adj.T.copy()
    td_ = np.where(td_ >= pos_threshold, 1.0, 0.0)
    tdt = np.matmul(td_, td_.T)
    tdt = tdt / tdt.sum(axis=-1).reshape(-1, 1)
    tdt = np.nan_to_num(tdt)
    tdt += np.eye(num_target, num_target)
    tdt = tdt.astype("float32")
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    tAll = tdt + t_t
    target_pos = np.zeros((num_target, num_target))
    for i in range(len(tAll)):
        one = tAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-tAll[i, one])
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = 1
        else:
            target_pos[i, one] = 1
    target_pos = sp.coo_matrix(target_pos)
    target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)

    if dataset == "davis":
        adj[adj != 0] -= 5
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T
    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)
    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj[train_row_ids, train_col_ids]
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj)
    adj_features[adj != 0] = 1
    features = np.concatenate((node_type_features, adj_features), 1)
    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph, drug_pos, target_pos


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)
    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))

from scipy.stats import pearsonr
def pearson(y, f):
    rp, p_value = pearsonr(y, f)
    return rp

def model_evaluate(Y, P):
    return [get_mse(Y, P),
            concordance_index(Y, P),
            get_rm2(Y, P),
            pearson(Y, P)]
