import os
import json
import torch
import random
import numpy as np
import scipy.io as io
from PIL import Image
from torch.utils.data.dataset import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class COCO2014_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path+'/'+self.X[index]).convert('RGB')
        
        if self.random_crops==0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)
        
        y = self.Y[index]

        return x, y, index
        # return x, y

    def __len__(self):
        return len(self.X)

class COCO2014_handler_two_augment(Dataset):
	def __init__(self, X, Y, data_path, transform_1=None, transform_2=None, random_crops=0):
		self.X = X
		self.Y = Y
		self.transform1 = transform_1
		self.transform2 = transform_2
		self.random_crops = random_crops
		self.data_path = data_path

	def __getitem__(self, index):
		x = Image.open(self.data_path+'/'+self.X[index]).convert('RGB')
		
		if self.random_crops == 0:
			x_1 = self.transform1(x)
			x_2 = self.transform2(x)
		else:
			crops = []
			for i in range(self.random_crops):
				crops.append(self.transform(x))
			x = torch.stack(crops)
		
		y = self.Y[index]

		return x_1, x_2, y, index

	def __len__(self):
		return len(self.X)

def get_COCO2014(train_data_path, test_data_path):
	
	img_list = json.load(open(train_data_path, 'r'))
	names = []
	labels = []

	for i in range(len(img_list)):
		item = img_list[i]
		names.append(item['file_name'])

		tmp_idxs = item['labels']
		lbl = np.zeros(80)
		lbl[tmp_idxs] = 1
		labels.append(lbl)
	
	names = np.array(names)
	labels = np.array(labels)

	rand_idxs = np.random.permutation(names.shape[0])
	names = names[rand_idxs]
	labels = labels[rand_idxs]

	train_data = names
	train_labels = labels.astype(np.float)

	img_list = json.load(open(test_data_path, 'r'))
	names = []
	labels = []

	for i in range(len(img_list)):
		item = img_list[i]
		names.append(item['file_name'])

		tmp_idxs = item['labels']
		lbl = np.zeros(80)
		lbl[tmp_idxs] = 1
		labels.append(lbl)
	
	names = np.array(names)
	labels = np.array(labels)
	
	rand_idxs = np.random.permutation(names.shape[0])
	names = names[rand_idxs]
	labels = labels[rand_idxs]

	test_data = names
	test_labels = labels.astype(np.float)

	return train_data, train_labels, test_data, test_labels

def generate_noisy_labels(labels, noise_level=0.05):

	N, C = labels.shape
		
	alpha_mat = np.ones_like(labels) * noise_level
	rand_mat = np.random.rand(N, C)

	mask = np.zeros((N, C), dtype=np.float)
	mask[labels != 1] = rand_mat[labels != 1] < alpha_mat[labels != 1]
	
	plabels = labels.copy()
	# plabels[mask==1] = -plabels[mask==1]
	plabels[mask==1] = 1

	return plabels
