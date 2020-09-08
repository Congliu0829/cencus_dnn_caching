from autoencoder import AutoEncoder 
import torch	
ae1 = AutoEncoder(10, 2)
ae2 = AutoEncoder(5, 2)
def convert(tensor_from_layer1, tensor_from_layer2, tensor_labels):
	'''This function aims to convert tensors to list (single data points)'''
	a1 = []
	a2 = []
	b1 = []
	b2 = []
	all_lst1 = []
	all_lst2 = []
	each_label = []
	for tensor1 in tensor_from_layer1:
		x_out1 = ae1.encode(tensor1)
		a1.append(x_out1.detach().numpy()[0][0])
		b1.append(x_out1.detach().numpy()[0][1])
		all_lst1.append([x_out1.detach().numpy()[0][0], x_out1.detach().numpy()[0][1]])
	
	for tensor2 in tensor_from_layer2:
		x_out2 = ae2.encode(tensor2)
		a2.append(x_out2.detach().numpy()[0][0])
		b2.append(x_out2.detach().numpy()[0][1])
		all_lst2.append([x_out2.detach().numpy()[0][0], x_out2.detach().numpy()[0][1]])

	for tensor_l in tensor_labels:
		each_label.append(tensor_l.numpy()[0][0])

	return a1, b1, all_lst1, a2, b2, all_lst2, each_label


