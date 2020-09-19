def get_label(layer_num, use_cluster, box_num, l1_first, l1_second, l2_first, l2_second, test_layer, cache_label, centroid, confidence, percentage, threshold):
	#compute epsilon
	per = 0.0000001
	e_a1 = (max(l1_first) - min(l1_first))*percentage*per
	e_b1 = (max(l1_second) - min(l1_second))*percentage*per
	e_a2 = (max(l2_first) - min(l2_first))*percentage*per
	e_b2 = (max(l2_second) - min(l2_second))*percentage*per

	if layer_num == 1:
		e_a = e_a1
		e_b = e_b1
	else:
		e_a = e_a2
		e_b = e_b2	

	test_label = []
	neighbor = {i:[] for i in range(len(test_layer))}
	for i in range(len(test_layer)):
		confidence1 = 0
		confidence0 = 0
		for j in range(len(centroid)): 	
			#find the intersection of first and second element in centroid_list
			if (test_layer[i][0] - e_a) <= centroid[j][0] and centroid[j][0] <= (test_layer[i][0] + e_a):
				if (test_layer[i][1] - e_b) <= centroid[j][1] and centroid[j][1] <= (test_layer[i][1] + e_b):
					neighbor[i].append(['value:',centroid[j], 'label:', cache_label[j]])
					if use_cluster == True:
						#when use cluster
						if j < box_num:
							confidence0 += confidence[j]	
						if j >= box_num:
							confidence1 += confidence[j]
					else:
						#For not use cluster
					    if cache_label[j] == 0:
						    confidence0 += confidence[j] + 1	
					    else:
						    confidence1 += confidence[j]
		if (confidence1 + confidence0) == 0:
		    label = 'do not have neighbor in this range'
		else:
		    confidence_t = confidence0 + confidence1
		    confidence0 /= confidence_t
		    confidence1 = 1 - confidence0
		if confidence0 > threshold:
		    label = '0'
		elif confidence1 > threshold:
		    label = '1'
		else:
			label = 'not sure'

		test_label.append(label)

	return test_label, neighbor

