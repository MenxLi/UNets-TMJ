from matplotlib import pyplot as plt
import numpy as np

def plotLR(lr_instance, max_epoch = 1000):
	lr = [lr_instance(i, max_epoch, 1) for i in range(max_epoch)]
	plt.plot(lr)
	plt.show()