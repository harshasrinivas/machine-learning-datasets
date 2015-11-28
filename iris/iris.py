import numpy as np
import pylab, sys
filename = 'iris.data'
alpha = 0.1
iterations = 4000
n = 5
m = 150


# Sigmoid Function

def sigmoid(z):
	return 1/(1+np.exp(-1*z))


# Cost Function

def cost(x, y, theta):
	term1 = np.dot(y.T, np.log(sigmoid(np.dot(theta.T, x.T))).T)
	term2 = np.dot((1-y).T, np.log(1 - sigmoid(np.dot(theta.T, x.T))).T)
	return np.sum(term1+term2)*(-1/m)


# Gradient descent calculation

def gradient(x, y, theta):
	J = np.zeros(iterations)
	for i in range(0,iterations):
		theta = theta - ( (alpha/m) * np.dot( (sigmoid(np.dot(theta.T, x.T)).T \
                                         - y).T , x ).T )
		J[i] = cost(x, y, theta)
	#plot(J)
	return theta


# Plot variation of Cost Function with each iteration

def plot(J):
	i = np.arange(0, iterations, 1);
	pylab.plot(i, J)
	pylab.show()


def main():

	x = np.loadtxt(filename, delimiter=',', usecols=[0,1,2,3])
	x = np.c_[np.ones(m), x]

	classify = np.loadtxt(filename, delimiter=',', usecols=[4], dtype='|S15')

	theta = np.zeros(n)[np.newaxis].T
	theta1 = np.zeros(n)[np.newaxis].T
	theta2 = np.zeros(n)[np.newaxis].T
	theta3 = np.zeros(n)[np.newaxis].T

	y1 = np.zeros(m)[np.newaxis].T
	y2 = np.zeros(m)[np.newaxis].T
	y3 = np.zeros(m)[np.newaxis].T

	for i in range(0,m):
		if classify[i] == 'Iris-setosa':
			y1[i]=1
		if classify[i] == 'Iris-versicolor':
			y2[i]=1
		if classify[i] == 'Iris-virginica':
			y3[i]=1

	theta1 = gradient(x, y1, theta)
	theta2 = gradient(x, y2, theta)
	theta3 = gradient(x, y3, theta)

	sample = np.array([1.0,float(sys.argv[1]),float(sys.argv[2]),\
									float(sys.argv[3]),float(sys.argv[4])])
	
	# probability of these 3 classes

	setosa = sigmoid(np.dot(theta1.T, sample))
	versicolor = sigmoid(np.dot(theta2.T, sample))
	virginica = sigmoid(np.dot(theta3.T, sample))

	probability = max(setosa, versicolor, virginica)
	
	if probability == setosa:
		print 'Species - Iris Setosa'
	if probability == versicolor:
		print 'Species - Iris Versicolor'
	if probability == virginica:
		print 'Species - Iris Virginica'

main()
