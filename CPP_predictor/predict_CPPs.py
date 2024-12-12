import numpy as np
import torch
from CPP_model import CPP_model
from utils import *
import pickle
import pandas as pd

max_len = 52 # maximun peptide length
word2idx, idx2word = make_vocab()
emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx)
vocab_size = len(word2idx)
emb_size = np.shape(emb)[1]


#Load Top CPP prediction models to do ensemble learning
repeat_num = 5
key_list = [
'3_0.0001_64',
'3_0.0001_128',
'3_1e-05_128',
'2_1e-05_64',
'2_1e-06_128'
]


all_list = []
ensemble_num = repeat_num * len(key_list)

deep_model_list = []
for a_key in key_list:
  for a_en in range(repeat_num):
    name = './models/trained_cpp_model_'+a_key+'_ensemble_'+str(a_en)
    model = torch.load('./'+name)
    deep_model_list.append(model)


def CPP_scoring(seq_list):
	seq_list = np.array(seq_list)

	ensemble_counter = 0
	for ensemble_id in range(ensemble_num):

		CPP_model = deep_model_list[ensemble_id].cuda()

		data_len = len(seq_list)
		batch_size = 8000
		for i in range(int(math.ceil(data_len/float(batch_size)))):
			if (i*batch_size) % 1000 == 0:
				print ('progress', i*batch_size, data_len)

			seq_batch = seq_list[i*batch_size:(i+1)*batch_size]
			seq_rep, _, _ = onehot_encoding(seq_batch, max_len, word2idx)

			X_seq = torch.LongTensor(seq_rep).cuda()

			CPP_pred_batch = F.sigmoid(CPP_model.cls_forward(X_seq)).cpu().detach().numpy()

			if i == 0:
				CPP_pred = CPP_pred_batch
			else:
				CPP_pred = np.vstack([CPP_pred, CPP_pred_batch])

		if ensemble_id == 0:
			CPP_sum = CPP_pred
		else:
			CPP_sum += CPP_pred
		ensemble_counter += 1

	CPP_pred = CPP_sum / float(ensemble_counter)

	df = pd.DataFrame(data=CPP_pred, columns=['CPP probability'], index=seq_list)
	print (df)

	df.to_csv('CPP_prediction.csv')

example_list = ['RRRRRRRRRR','LLLLLLLLLLLLLLLL']
CPP_scoring(example_list)
