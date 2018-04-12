import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import pickle
import numpy as np




def covariance_matrix(df):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(np.cov(np.transpose(df)), interpolation="nearest", cmap=cmap)
    #ax1.grid(True)
    plt.title('Covariance Matrix ')
    #plt.axis('off')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


#Loading the data
X=[]
y=[]
with open('wifi_localization.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	for row in datareader:
		#print map(int,row[:7])
		X.append(map(int,row[:7]))
		y.append(int(row[7]))

#print len(X),len(y)
with open('y.pkl', 'wb') as f:
	pickle.dump(y, f)

#Dimensionality reduction to 2 dimensions with LDA
#lda = LinearDiscriminantAnalysis(n_components=2)
#X_reduced_2 = lda.fit(X, y).transform(X)
#with open('X_reduced_2.pkl', 'wb') as f:
#	pickle.dump(X_reduced_2, f)

#Dimensionality reduction to 2 dimensions with LDA
lda = LinearDiscriminantAnalysis(solver="eigen",n_components=2)
X_reduced_2 = lda.fit(X, y).transform(X)

covariance_matrix(X)
covariance_matrix(X_reduced_2)

#with open('X_reduced_2.pkl', 'wb') as f:
#	pickle.dump(X_reduced_2, f)

#Dimensionality reduction to 1 dimension with LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_reduced_1 = lda.fit(X, y).transform(X)
#with open('X_reduced_1.pkl', 'wb') as f:
#	pickle.dump(X_reduced_1, f)

#print X_reduced_2
#print X_reduced_1

#Plotting the reduced dataset to visualize 2d patterns
for i in range(2000):
	if y[i]==1:
		plt.plot(X_reduced_2[i,0], X_reduced_2[i,1], 'ro')
	elif y[i]==2:
		plt.plot(X_reduced_2[i,0], X_reduced_2[i,1], 'go')
	elif y[i]==3:
		plt.plot(X_reduced_2[i,0], X_reduced_2[i,1], 'bo')
	elif y[i]==4:
		plt.plot(X_reduced_2[i,0], X_reduced_2[i,1], 'yo')

plt.show()

'''
#Plotting the reduced dataset to visualize 1d patterns
for i in range(2000):
	if y[i]==1:
		plt.plot(X_reduced_1[i,0], 'ro')
	elif y[i]==2:
		plt.plot(X_reduced_1[i,0], 'go')
	elif y[i]==3:
		plt.plot(X_reduced_1[i,0], 'bo')
	elif y[i]==4:
		plt.plot(X_reduced_1[i,0], 'yo')

plt.show()'''


