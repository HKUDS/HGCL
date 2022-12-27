import numpy as np 
import scipy.sparse as sp

import torch.utils.data as data

import pickle

class BPRData(data.Dataset):
	def __init__(self, data, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.data = np.array(data)
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'
		tmp_trainMat = self.train_mat.todok()
		length = self.data.shape[0]
		self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)

		for i in range(length):
			uid = self.data[i][0]
			iid = self.neg_data[i]
			if (uid, iid) in tmp_trainMat:
				while (uid, iid) in tmp_trainMat:
					iid = np.random.randint(low=0, high=self.num_item)
				self.neg_data[i] = iid

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		user = self.data[idx][0]
		item_i = self.data[idx][1]
		if self.is_training:
			neg_data = self.neg_data
			item_j   = neg_data[idx]
			return user, item_i, item_j 
		else:
			return user, item_i
		