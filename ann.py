import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn import preprocessing

with open('X_reduced_2_train.pkl', 'rb') as f:
	X_reduced_2_train = pickle.load(f)
with open('X_reduced_2_test.pkl', 'rb') as f:
	X_reduced_2_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
	y_train = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
	y_test = pickle.load(f)

lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
print lb.classes_

y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

trained_mlp=MLPClassifier(hidden_layer_sizes=(5,6,4 ),max_iter=10000,verbose=True,solver="adam").fit(X_reduced_2_train,y_train)
print trained_mlp.score(X_reduced_2_test,y_test)
