import os
import argparse
import torch
import json

import torch_geometric
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from collections import OrderedDict
from torch import nn
from itertools import chain
from create_data import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph, \
    get_clique_molecule_graph
from utils_data import *
from models_TCCL import TCCLDTA, PredictModule


def train(model, predictor, device, optimizer, train_loader, affinity_graph, drug_seq, target_seq, drug_graphs_DataLoader,
          clique_graphs_DataLoader,
          target_graphs_DataLoader, epoch, drug_pos, target_pos, args):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 20
    loss_fn = nn.MSELoss()

    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    clique_graph_batchs = list(map(lambda graph: graph.to(device), clique_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph, drug_seq, target_seq, drug_graph_batchs,
                                                           clique_graph_batchs, target_graph_batchs, drug_pos,
                                                           target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * args.batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def predict(model, predictor, device, test_loader, affinity_graph, drug_seq, target_seq, drug_graphs_DataLoader,
            clique_graphs_DataLoader, target_graphs_DataLoader, drug_pos, target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))

    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    clique_graph_batchs = list(map(lambda graph: graph.to(device), clique_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))

    with torch.no_grad():
        for data in test_loader:
            _, drug_embedding, target_embedding = model(affinity_graph, drug_seq, target_seq, drug_graph_batchs,
                                                        clique_graph_batchs, target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))

    dataset_path = f'/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/{args.dataset}/'
    train_data, test_data, affinity_graph, drug_pos, target_pos = positive_create(args.dataset,
                                                                                  dataset_path, args.num_pos,
                                                                                  args.pos_threshold)

    drug_smile_list = []
    protein_seq_list = []
    smile = json.load(
        open(f'/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/{args.dataset}/drugs.txt'))
    # print(smile)
    seq = json.load(open(f'/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/{args.dataset}/targets.txt'))
    for k, v in smile.items():
        drug_smile = torch.from_numpy(label_smiles(v, CHARISOSMISET, 100))
        drug_smile_list.append(drug_smile)
    for k, v in seq.items():
        protein_seq = torch.from_numpy(label_sequence(v, CHARPROTSET, 1200))
        protein_seq_list.append(protein_seq)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    drug_seq_Data = CustomDataSet(drug_smile_list)
    drug_seq_DataLoader = DataLoader(drug_seq_Data, shuffle=False, collate_fn=collate,
                                     batch_size=affinity_graph.num_drug)
    target_seq_Data = CustomDataSet(protein_seq_list)
    target_seq_DataLoader = DataLoader(target_seq_Data, shuffle=False, collate_fn=collate,
                                       batch_size=affinity_graph.num_target)
    print('building drug graph.......')
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/{args.dataset}/drugs.txt'),
                  object_pairs_hook=OrderedDict))
    print('building clique graph.......')
    clique_graphs_dict = get_clique_molecule_graph(
        json.load(open(f'/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/{args.dataset}/drugs.txt'),
                  object_pairs_hook=OrderedDict))

    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch_geometric.loader.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                               batch_size=affinity_graph.num_drug)
    clique_graphs_Data = GraphDataset(graphs_dict=clique_graphs_dict, dttype="drug")
    clique_graphs_DataLoader = torch_geometric.loader.DataLoader(clique_graphs_Data, shuffle=False, collate_fn=collate,
                                                                 batch_size=affinity_graph.num_drug)
    print('building target graph.......')
    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'/home/july/PycharmProjects/23AIBox-CSCo-DTA-main/data/data/{args.dataset}/targets.txt'),
                  object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch_geometric.loader.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                                 batch_size=affinity_graph.num_target)
    model_file_name = f'results_lr/results_lr1/{args.dataset}/' + f'best_model.model'
    result_file_name = f'results_lr/results_lr1/{args.dataset}/' + f'best_model.csv'
    model_file_name2 = f'results_lr/results_lr1/{args.dataset}/' + f'pre_model.model'
    model_file = f'results_lr/results_lr1/{args.dataset}/'
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    ns_dims = [affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256]
    model = TCCLDTA(args.tau, args.lam, ns_dims, args.edge_dropout_rate, embedding_dim=128)
    predictor = PredictModule()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=args.lr,
        weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=100,
                                                           verbose=True, min_lr=3e-4)

    path1 = '/home/july/PycharmProjects/ASGDTA-main/model_affinity/results_sage/results_sage7/kiba/best_model1228.model'
    path2 = '/home/july/PycharmProjects/ASGDTA-main/model_affinity/results_sage/results_sage7/kiba/pre_model1228.model'
    check_point = torch.load(path1, map_location=device)  # Load the model parameters trained on the KIBA dataset.
    check_point2 = torch.load(path2, map_location=device)
    model.load_state_dict(check_point)  # Loading pre-trained model parameters into the current model.
    predictor = PredictModule()
    predictor.load_state_dict(check_point2)
    model.to(device)
    predictor.to(device)
    # drug_pos.to(device)
    # target_pos.to(device)
    best_mse = 100
    best_ci = 0
    best_epoch = 0
    print("Start training...")
    for epoch in range(args.epochs):
        train(model, predictor, device, optimizer, train_loader, affinity_graph.to(device), drug_seq_DataLoader, target_seq_DataLoader,
              drug_graphs_DataLoader,
              clique_graphs_DataLoader, target_graphs_DataLoader,
              epoch, drug_pos, target_pos, args)
        G, P = predict(model, predictor, device, test_loader, affinity_graph.to(device), drug_seq_DataLoader,
                       target_seq_DataLoader, drug_graphs_DataLoader, clique_graphs_DataLoader,
                       target_graphs_DataLoader, drug_pos, target_pos)

        ret = model_evaluate(G, P)
        scheduler.step((ret[0]))
        if ret[0] < best_mse:
            with open(result_file_name, 'a') as f:
                f.write('epoch' + str(epoch) + ' ' + ','.join(map(str, ret)) + '\n')
            best_epoch = epoch + 1
            best_mse = ret[0]
            best_ci = ret[1]
            # if epoch > 1000:
            #     model_file_name = f"results_sage/results_sage1/{args.dataset}/best_model{str(epoch)}.model"
            #     model_file_name2 = f"results_sage/results_sage1/{args.dataset}/pre_model{str(epoch)}.model"
            #     torch.save(model.state_dict(), model_file_name)
            #     torch.save(predictor.state_dict(), model_file_name2)
            # else:
            torch.save(model.state_dict(), model_file_name)
            torch.save(predictor.state_dict(), model_file_name2)
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, args.dataset)
        else:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,
                  args.dataset)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='kiba')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=10)
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    train_predict()
