''' This function aims to split elements in embeddings to size 1, for batchsize = 1'''
import torch
def split(data, store_list):
	for i in range(len(data)):
		tupl = torch.split(data[i],split_size_or_sections=1,dim = 0)#use a tuple to store 
		for i in range(len(list(tupl))):
			store_list.append(list(tupl)[i])
	return store_list
