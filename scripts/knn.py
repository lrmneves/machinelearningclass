import numpy as np
from sklearn import datasets
import math
def euclideanDistance(x,y):
	aggregator = 0.0

	for i in range(len(x)):
		aggregator += (x[i] - y[i])**2

	return math.sqrt(aggregator) if math.sqrt(aggregator) != 0 else 100

def getKNN(instance,data,k):
	neighbours = []
	for other_instance_index in range(len(data)):
		other_instance = data[other_instance_index]
		neighbours.append((other_instance,other_instance_index,euclideanDistance(instance,other_instance)))
	neighbours.sort(key=lambda x: x[2])
	# print neighbours[:k]
	return neighbours[:k]
def getClass(instance,neighbours,dataset_class,num_classes):
	classes = [0]*num_classes
	max_v = 0
	max_i = 0
	for n in neighbours:
		classes[dataset_class[n[1]]]+=1
	for i in range(len(classes)):
		if(classes[i] > max_v):
			max_v = classes[i]
			max_i = i
	return max_i


iris = datasets.load_iris()
iris_data = iris.data
iris_class = iris.target
class_set = set(iris_class)


accuracy = 0
for i in range(len(iris_data)):
	instance = iris_data[i]
	i_class = getClass(instance,getKNN(instance,iris_data,10),iris_class,len(class_set))

	if(iris_class[i] == i_class):
		accuracy+=1

print accuracy*1.0/len(iris_data)



