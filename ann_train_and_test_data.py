import numpy as np
import pickle
import random

#Load pickle files
with open('X_reduced_2.pkl', 'rb') as f:
	X_reduced_2 = pickle.load(f)
with open('y.pkl', 'rb') as f:
	y = pickle.load(f)

X_reduced_2_class1 = X_reduced_2[0:500]
X_reduced_2_class2 = X_reduced_2[500:1000]
X_reduced_2_class3 = X_reduced_2[1000:1500]
X_reduced_2_class4 = X_reduced_2[1500:2000]

y_class1 = y[0:500]
y_class2 = y[500:1000]
y_class3 = y[1000:1500]
y_class4 = y[1500:2000]

dummy1 = list(zip(X_reduced_2_class1, y_class1))
random.shuffle(dummy1)
X_reduced_2_class1, y_class1 = zip(*dummy1)
dummy2 = list(zip(X_reduced_2_class2, y_class2))
random.shuffle(dummy2)
X_reduced_2_class2, y_class2 = zip(*dummy2)
dummy3 = list(zip(X_reduced_2_class3, y_class3))
random.shuffle(dummy3)
X_reduced_2_class3, y_class3 = zip(*dummy3)
dummy4 = list(zip(X_reduced_2_class4, y_class4))
random.shuffle(dummy4)
X_reduced_2_class4, y_class4 = zip(*dummy4)

X_reduced_2 = X_reduced_2_class1+X_reduced_2_class2+X_reduced_2_class3+X_reduced_2_class4
y = y_class1+y_class2+y_class3+y_class4

X_reduced_2_train=[]
X_reduced_2_test=[]
y_train=[]
y_test=[]

for i in range(0,2000,500):
	for j in range(0,500):
		if j<375:
			X_reduced_2_train.append(X_reduced_2[i+j])
			y_train.append(y[i+j])
		elif j>=375:
			X_reduced_2_test.append(X_reduced_2[i+j])
			y_test.append(y[i+j])

print len(y_train),len(y_test)

with open('X_reduced_2_train.pkl', 'wb') as f:
	pickle.dump(X_reduced_2_train, f)
with open('X_reduced_2_test.pkl', 'wb') as f:
	pickle.dump(X_reduced_2_test, f)
with open('y_train.pkl', 'wb') as f:
	pickle.dump(y_train, f)
with open('y_test.pkl', 'wb') as f:
	pickle.dump(y_test, f)		


