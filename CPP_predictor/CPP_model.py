import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class PeptideEmbeddings(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.aa_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb), padding_idx=0)
    def forward(self, x):
        out = self.aa_embedding(x)
        return out

class CPP_model(nn.Module):
    def __init__(self, emb, emb_size, num_rnn_layers, dim_h, dim_latent):
        super().__init__()

        self.peptideEmb = PeptideEmbeddings(emb=emb)
        self.dim_emb = emb_size
        self.dim_h = dim_h
        self.dropout = 0.1
        self.dim_latent = dim_latent
        max_len = 52

        self.rnn = nn.GRU(emb_size, dim_h, num_layers=num_rnn_layers, batch_first=True, dropout=0.1, bidirectional=True)#nn.LSTM(emb_size, dim_h, num_layers=num_rnn_layers, batch_first=True, dropout=0.1, bidirectional=True)#
        self.layernorm = nn.LayerNorm(dim_h * 2)
        self.attn1 = nn.Linear(dim_h * 2 + emb_size, max_len)
        self.attn2 = nn.Linear(dim_h * 2, 1)

        self.fc0 =  nn.Linear(dim_h * 2, dim_latent)
        self.fc1 =  nn.Linear(dim_latent, 1)

    def cls_forward(self, x):

        x = self.peptideEmb(x)
        out, h = self.rnn(x)
        out = self.layernorm(out)

        attn_weights1 = F.softmax(self.attn1(torch.cat((out, x), 2)), dim=2) #to be tested: masked softmax
        attn_weights1.permute(0, 2, 1)
        out = torch.bmm(attn_weights1, out)
        attn_weights2 = F.softmax(self.attn2(out), dim=1) #to be tested: masked softmax
        out = torch.sum(attn_weights2 * out, dim=1) #to be test: masked sum

        out = self.fc0(out)
        out = self.fc1(F.selu(out))

        return out

