import matplotlib.pyplot as plt

colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
def SubGraph(s1, s2, loc):
	return int(str(s1) + str(s2) + str(loc))

def PlotOneLine(x, y, color, subId=None):
	if subId != None:
		ax = plt.subplot(subId)
	if color:
		plt.plot(x, y, color)
	else:
		plt.plot(x, y)
	plt.show()

def PlotLineChart(x, y, xName = '', yName = '', subGraph=True):
	if x.shape != y.shape or len(x.shape) > 2:
		print(x.shape, y.shape)
		print('Input Data Error for Plotter')
		return
	plt.figure(1)
	if subGraph:
		if len(x.shape) == 2:
			for i in range(x.shape[0]):
				subId = SubGraph(x.shape[0], 1, i + 1)
				PlotOneLine(x[i], y[i], subId)
		else:
			subId = SubGraph(1, 1, 1)
			PlotOneLine(x, y, subId)
	else:
		if len(x.shape) == 2 and x[0] > len(colors):
			print('Too Many Curve, Use SubGraph')
			return
		for i in range(x.shape[0]):
			PlotOneLine(x[i], y[i], colors[i])
	plt.show()
