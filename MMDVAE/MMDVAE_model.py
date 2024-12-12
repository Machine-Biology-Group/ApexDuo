import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class PeptideVAE(nn.Module):
    """
    Variational peptide autoencoder based on RNN and attention.
    """
    def __init__(self, params={}):
        """
        Arguments:
            params (dict): dictionary of model parameters. params keys:
                l_encoder: num of encoder layers
                l_decoder: num of decoder layers
                l_pp: num of property prediction layers
                dim_emb: dimension of embeddings
                dim_h: dimension of RNN and attention
                dim_latent: dimension of latent vectors
                dim_pp: dimension of property prediction layers
                dropout: dropout rate
                pp: predict properties or not
                num_p: num of properties
                max_len: max length of input sequence 
                vocab_size: vocabulary size of input sequence
        """
        super().__init__()

        self.params = params

        self.peptideEmb = PeptideEmbeddings(vocab_size=params['vocab_size'], dim_emb=params['dim_emb'], dim_h=params['dim_h'])

        self.encoder = PeptideEncoder(peptideEmb=self.peptideEmb,  
                                      dim_emb=params['dim_emb'], dim_h=params['dim_h'] ,l_encoder=params['l_encoder'], 
                                      dropout=params['dropout'], max_len=params['max_len'], dim_latent=params['dim_latent'])

        self.decoder = PeptideDecoder(peptideEmb=self.peptideEmb, 
                                      dim_latent=params['dim_latent'], dim_h=params['dim_h'], l_decoder=params['l_decoder'], 
                                      dropout=params['dropout'], max_len=params['max_len'])

        self.generator = Generator(dim_h=params['dim_h'], vocab_size=params['vocab_size'])

        if params['pp']:
            self.property_predictor = PropertyPredictor(dim_latent=params['dim_latent'], l_pp=params['l_pp'], 
                                                        dim_pp=params['dim_pp'], num_p=params['num_p'])
        else:
            self.property_predictor = None


    def forward(self, src, src_mask=None):
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(encoder_out)
        src_rec = self.generator(decoder_out)
        if self.params['pp']:
            prop = self.property_predictor(encoder_out)
        else:
            prop = None
        return src_rec, encoder_out, prop

    def rand_sample(self, num_sample):
        sampled_codes = torch.randn((num_sample, self.params['dim_latent'])).cuda()
        decoder_out = self.decoder(sampled_codes)
        seq = self.generator(decoder_out)
        return sampled_codes, seq
    
    def predict_property_from_seq(self, x):
        encoder_out = self.encoder(x)
        prop = self.property_predictor(encoder_out)
        return prop
    
    def predict_property_from_latent(self, x):
        prop = self.property_predictor(x)
        return prop
    
    def decode_from_latent(self, x):
        decoder_out = self.decoder(x)
        seq = self.generator(decoder_out)
        return seq
    
    def decode_from_seq(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        src_rec = self.generator(decoder_out)
        return src_rec



class PeptideEmbeddings(nn.Module):
    def __init__(self, vocab_size, dim_emb, dim_h):
        super().__init__()
        self.peptideEmb = nn.Embedding(vocab_size, dim_emb, padding_idx=0)
        self.dim_h = dim_h

    def forward(self, x):
        return self.peptideEmb(x) * math.sqrt(self.dim_h)


class PeptideEncoder(nn.Module):
    def __init__(self, peptideEmb, dim_emb, dim_h ,l_encoder, dropout, max_len, dim_latent):
        super().__init__()

        self.peptideEmb = peptideEmb
        self.dim_emb = dim_emb
        self.dim_h = dim_h
        self.l_encoder = l_encoder
        self.dropout = dropout
        self.max_len = max_len
        self.dim_latent = dim_latent

        self.rnn = nn.GRU(dim_emb, dim_h, num_layers=l_encoder, batch_first=True, dropout=dropout, bidirectional=True)
        self.layernorm = nn.LayerNorm(dim_h * 2)
        self.attn1 = nn.Linear(dim_h * 3, max_len)
        self.attn2 = nn.Linear(dim_h * 2, 1)
        self.fc1 = nn.Linear(dim_h * 2, dim_latent)

    def forward(self, x):
        x = self.peptideEmb(x)
        h = self.initH(x.shape[0])
        out, h = self.rnn(x, h)
        out = self.layernorm(out)
        attn_weights1 = F.softmax(self.attn1(torch.cat((out, x), 2)), dim=2)
        attn_weights1.permute(0, 2, 1)
        out = torch.bmm(attn_weights1, out)
        attn_weights2 = F.softmax(self.attn2(out), dim=1)
        out = torch.sum(attn_weights2 * out, dim=1)
        out = self.fc1(out)
        return out 

    def initH(self, batch_size):
        h = np.ones((1,1)).astype(int)
        h = torch.LongTensor(h).cuda()
        h = self.peptideEmb(h)
        h = h.repeat(2 * self.l_encoder, batch_size, 1)
        return h 

class PeptideDecoder(nn.Module):
    def __init__(self, peptideEmb, dim_latent, dim_h, l_decoder, dropout, max_len):
        super().__init__()

        self.peptideEmb = peptideEmb
        self.dim_latent = dim_latent
        self.dim_h = dim_h
        self.l_decoder = l_decoder
        self.dropout = dropout
        self.max_len = max_len

        self.rnn = nn.GRU(dim_latent, dim_h, num_layers=l_decoder, batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim_h)

    def forward(self, x):
        h = self.initH(x.shape[0])
        x = x.unsqueeze(1).repeat(1, self.max_len, 1)
        out, h = self.rnn(x, h)
        out = self.layernorm(out)
        return out

    def initH(self, batch_size):
        h = np.ones((1,1)).astype(int)
        h = torch.LongTensor(h).cuda()
        h = self.peptideEmb(h)
        h = h.repeat(self.l_decoder, batch_size, 1)
        return h 

class Generator(nn.Module):
    def __init__(self, dim_h, vocab_size):
        super().__init__()
        self.proj = nn.Linear(dim_h, vocab_size)

    def forward(self, x):
        return self.proj(x)

class PropertyPredictor(nn.Module):
    def __init__(self, dim_latent, l_pp, dim_pp, num_p):
        super().__init__()
        prediction_layers = []
        if l_pp == 0:
            linear_layer = nn.Linear(dim_latent, num_p)
            prediction_layers.append(linear_layer)
        elif l_pp == 1:
            linear_layer = nn.Linear(dim_latent, dim_pp)
            prediction_layers.append(linear_layer)
            linear_layer = nn.Linear(dim_pp, num_p)
            prediction_layers.append(linear_layer)
        else:
            for i in range(l_pp):
                if i == 0:
                    linear_layer = nn.Linear(dim_latent, dim_pp)
                elif i == l_pp - 1:
                    linear_layer = nn.Linear(dim_pp, num_p)
                else:
                    linear_layer = nn.Linear(dim_pp, dim_pp)
                prediction_layers.append(linear_layer)
        self.prediction_layers = ListModule(*prediction_layers)

    def forward(self, x):
        for prediction_layer in self.prediction_layers:
            x = F.relu(prediction_layer(x))
        return x

class ListModule(nn.Module):
    """Create single pytorch module from list of modules"""
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

