import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
print(lb.classes_)

y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

trained_mlp=MLPClassifier(hidden_layer_sizes=(8),max_iter=1,verbose=False,solver="adam",alpha=0.4)
scores_train = []
scores_test = []
for i in range(2500):
	print(i)
	trained_mlp.partial_fit(X_reduced_2_train,y_train, np.array([1, 2, 3, 4]))
	trained_mlp.score(X_reduced_2_train, y_train)
	scores_train.append(trained_mlp.loss_)
	trained_mlp.score(X_reduced_2_test, y_test)
	scores_test.append(trained_mlp.loss_)

print(" Training Accuracy is " , trained_mlp.score(X_reduced_2_train,y_train))
print(" Testing Acuuracy is " , trained_mlp.score(X_reduced_2_test,y_test))

plt.figure(1)
plt.plot(scores_train)
plt.title("Loss on train Data")
plt.show()

plt.figure(2)
plt.plot(scores_test)
plt.title("Loss on test data")
plt.show()