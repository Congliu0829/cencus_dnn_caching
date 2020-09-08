from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_his_3d_both(data_ele1, data_ele2, data_ele3, data_ele4, width, depth, range1, range2, titl):
	fig = plt.figure()
	ax = fig.add_subplot(projection = '3d', title = titl)
	x = data_ele1
	y = data_ele2
	x1 = data_ele3
	y1 = data_ele4
	if len(data_ele1) > len(data_ele2):
		x = x[:len(y)]
	elif len(data_ele1) < len(data_ele2):
		y = y[:len(x)]
	hist, xedges, yedges = np.histogram2d(x, y, bins = 10, range = [[min(range1),max(range1)],[min(range2), max(range2)]])
	hist1, xedges, yedges = np.histogram2d(x1, y1, bins = 10, range = [[min(range1),max(range1)],[min(range2), max(range2)]])
	xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing = "ij")
	xpos = xpos.ravel()
	ypos = ypos.ravel()
	zpos = 0

	dx = width*np.ones_like(zpos)
	dy = depth*np.ones_like(zpos)
	dz = hist.ravel()
	dz1 = hist1.ravel()

	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort = 'average')
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz1, zsort = 'average')
	ax.set_xlabel('first element')
	ax.set_ylabel('second element')
	ax.set_zlabel('numbers')
	plt.show()
