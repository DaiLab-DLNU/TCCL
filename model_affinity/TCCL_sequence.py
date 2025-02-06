import torch
import torch.nn as nn
import torch.nn.functional as F


class Sequence_drug(nn.Module):
    def __init__(self, drug_MAX_LENGH=100, drug_kernel=[4, 6, 8],
                 conv=32, char_dim=100, head_num=8, dropout_rate=0.1):
        super(Sequence_drug, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel

        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(
            self.drug_MAX_LENGH - self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(96, 256)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        torch.nn.init.constant_(self.out.bias, 5)

    def forward(self, drug):
        drug = drug.to('cuda:0')
        drugembed = self.drug_embed(drug)
        # print(drugembed.device)
        drugembed = drugembed.permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)
        # print('drugConv', drugConv.size())
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        drugConv = self.leaky_relu(self.fc1(drugConv))
        # drugConv = self.dropout1(drugConv)
        # print('drugConv', drugConv.size())

        return drugConv


class Sequence_protein(nn.Module):
    def __init__(self, protein_MAX_LENGH=128, protein_kernel=[4, 8, 12],
                 conv=32, char_dim=128, head_num=8, dropout_rate=0.1):
        super(Sequence_protein, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        # self.Protein_max_pool = nn.MaxPool1d(
        #     self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.Protein_max_pool = nn.MaxPool1d(kernel_size=1179)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(96, 256)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        torch.nn.init.constant_(self.out.bias, 5)

        # # model_name = '/distilbert-base-uncased'
        # model_name = 'huawei-noah/TinyBERT_General_4L_312D'
        # # self.k_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.k_bert = BertModel.from_pretrained(model_name)
        # # k-bert
        # # model_path = './models/k_bert.pth'
        # # self.k_bert = torch.load(model_path)
        # # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, protein):
        protein = protein.to('cuda:0')
        proteinembed = self.protein_embed(protein)
        proteinembed = proteinembed.permute(0,2,1)
        proteinConv = self.Protein_CNNs(proteinembed)
        # print('proteinConv_before', proteinConv.size())
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        proteinConv = self.leaky_relu(self.fc1(proteinConv))
        # proteinConv = self.dropout1(proteinConv)
        # print('proteinConv', proteinConv.size())

        return proteinConv
