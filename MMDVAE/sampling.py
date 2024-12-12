import numpy as np
import torch
from MMDVAE_model import PeptideVAE
from Bio import SeqIO
import string
from sklearn.neighbors import KernelDensity
import pickle
import math

aa_list = []
aa_list.append('A')
aa_list.append('C')
aa_list.append('D')
aa_list.append('E')
aa_list.append('F')
aa_list.append('G')
aa_list.append('H')
aa_list.append('I')
aa_list.append('K')
aa_list.append('L')
aa_list.append('M')
aa_list.append('N')
aa_list.append('P')
aa_list.append('Q')
aa_list.append('R')
aa_list.append('S')
aa_list.append('T')
aa_list.append('V')
aa_list.append('W')
aa_list.append('Y')
aa_list.append('X')

max_len = 51 # maximun seq length

f = open('./Pretrained_word2idx_vocab', 'rb')
word2idx = pickle.load(f)
f.close()

f = open('./Pretrained_idx2word_vocab', 'rb')
idx2word = pickle.load(f)
f.close()

vocab_size = len(word2idx)


def isEnglish(s):
	#return s.translate(None, string.punctuation).isalnum()
	return s.translate(str.maketrans('','',string.punctuation)).isalnum()

def onehot_encoding(seq_list, max_len, word2idx):
	#2: end
	X = np.zeros((len(seq_list), max_len)).astype(int)


	for i in range(len(seq_list)):
		a_seq = seq_list[i].upper()+'2'
		if len(a_seq) > max_len:
			iter_num = max_len
		else:
			iter_num = len(a_seq)
		for j in range(iter_num):
			if a_seq[j] not in word2idx:
				continue
			else:
				X[i,j] = word2idx[a_seq[j]]
	return np.array(X)


def init_model(max_len, vocab_size):

	params = {}
	params['l_encoder'] = 3
	params['l_decoder'] = 3
	params['l_pp'] = 1
	params['dim_emb'] = 256
	params['dim_h'] = 256
	params['dim_latent'] = 128
	params['dim_pp'] = 64
	params['dropout'] = 0.1
	params['pp'] = True
	params['num_p'] = 30 #deprecated
	params['max_len'] = max_len
	params['vocab_size'] = vocab_size

	model = PeptideVAE(params)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print ('Model initialized, total num params:', pytorch_total_params)

	return model 


def load_model(checkpoint_path):

	model = init_model(max_len, vocab_size)
	checkpoint = torch.load(checkpoint_path)    

	model.load_state_dict(checkpoint['model_state_dict'], strict=False)
	print ('Model loaded')
	model.cuda() 
	model.eval()

	return model

def logtis2seq(logits, idx2word):
	seq_list = []
	for a_seq in logits:
		idx = np.argmax(a_seq, axis=1)
		seq_real = ''
		for k in idx:
			seq_real += idx2word[k]
			if idx2word[k] == '2':
				break
		seq_list.append(seq_real)
	return seq_list



model = load_model('./trainedMMDVAE').cuda()

def sample_from_KDE(seq_list, num_samples, save_file_name):
    """
    Arguments:
            seq_list: prior sequences; the function will fit a KDE based on the VAE latent codes of prior seqs; new latent codes will be sampled from the fitted KDE
			num_samples: number of samples you want to sample from the KDE
			save_file_name: the path where valid generated peptides will be saved 
    """
	seq_rep = onehot_encoding(seq_list, max_len, word2idx)

	data_len = len(seq_rep)

	batch_size = 500
	for i in range(int(math.ceil(data_len/float(batch_size)))):
		X_seq = seq_rep[i*batch_size:(i+1)*batch_size]
		X_seq = torch.LongTensor(X_seq).cuda()
		seq_rec_logits, latent_batch, _  = model(X_seq)
		if i == 0:
			latent_codes = latent_batch.cpu().detach().numpy()
		else:
			latent_codes = np.vstack([latent_codes, latent_batch.cpu().detach().numpy()])

	kde = KernelDensity(kernel='gaussian').fit(latent_codes)
	sampled_c = kde.sample(n_samples=num_samples)

	batch_size = 8000
	valid_list = []
	for i in range(int(math.ceil(len(sampled_c)/float(batch_size)))):


		X_lc = sampled_c[i*batch_size:(i+1)*batch_size]
		X_lc = torch.FloatTensor(X_lc).cuda()

		seq_rec_logits = model.decode_from_latent(X_lc)
		seq_rec_logits = seq_rec_logits.cpu().detach().numpy()
		seq_rec = logtis2seq(seq_rec_logits, idx2word)

		#keep valid seqs
		for j in seq_rec:
			if len(j) <= 1:
				continue

			if '2' not in j:
				continue
			if '2' in j[:-1]:
				continue
			if '0' in j:
				continue
			if 'X' in j:
				continue
			if 'U' in j:
				continue
			if 'Z' in j:
				continue
			if 'B' in j:
				continue

			valid_list.append(j[:-1])


	valid_set = set()
	for i in valid_list:
		valid_set.add(i)
	valid_list = [i for i in valid_set]

	f = open(save_file_name, 'w')
	for i in valid_list:
		f.writelines(i+'\n')
	f.close()


example_prior_seqs = ['RRRRRRRRRRR','LLLLLLLLLLLLLLLL']
sample_from_KDE(example_prior_seqs, 100, 'test.txt')
