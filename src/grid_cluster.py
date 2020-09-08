# This function aims to create clusters in cache, store confidence and centroid of each layer's output.
# return value with centroid and each confidence.
# Confidence is computed by adding each label's appearance
from sklearn.cluster import KMeans
import numpy as np

def grid_cluster(data, data_1, data_2, grid_num, num_cluster):
	max_1 = max(data_1)
	min_1 = min(data_1)
	max_2 = max(data_2)
	min_2 = min(data_2)
	interval_1 = (max_1 - min_1)/grid_num
	interval_2 = (max_2 - min_2)/grid_num

	data_box = {i: [] for i in range(1,grid_num*grid_num+1)}
	confidence_of_databox = []
	cluster_centers = []
	for index in range(len(data)):
		#找出每一行和每一列落在方块里面的数据。 横着走的，先分纵向，再分横向。
		for n in range(grid_num):
			for j in range(grid_num):
				if data[index][0]>= min_1 + n*interval_1 and data[index][0] < (n+1)*interval_1 + min_1 and data[index][1] >= j*interval_2 + min_2 and data[index][1] < (j+1)*interval_2 + min_2:
					data_box[(n+1)*(j+1)].append(data[index])

	for data in data_box.values():
		if data == []:
			confidence_of_databox.append(len(data))
			cluster_centers.append([0, 0])
		else:
		    confidence_of_databox.append(len(data))
		    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(data)
		    cluster_centers.append(kmeans.cluster_centers_[0])
	return confidence_of_databox, cluster_centers


