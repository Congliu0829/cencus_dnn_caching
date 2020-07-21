import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Histogram:
	def __init__(self, dataname, data, max_ele, min_ele):
		self.data = data
		self.dataname = dataname
		self.space = (max_ele - min_ele)/100
		self.bins = [i*self.space + min_ele for i in range(1,101)]


	def plot(self, data, dataname, bins):
		plt.hist(x = self.data, bins = self.bins, density = True)
		plt.grid(axis = 'y', alpha = 0.75)
		plt.xlabel('Value')
		plt.ylabel('Frequency')
		plt.title('The histogram of '+ dataname)


