import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

plt.xlabel("iterations")
plt.ylabel("Total Error")
plt.title("Error function ")
colors = [ 'b', 'c', 'r', 'm' ]

with open('plotData_.txt') as plotData:

	data_points = eval(plotData.read())


for i,points in enumerate(data_points):
	color = colors[points[0]]
	# print points[1]
	plt.plot(i, points[1], color + "+" )

plt.show()