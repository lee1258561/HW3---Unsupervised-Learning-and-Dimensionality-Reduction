import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from time import time
from utils import * 

def run_NN(X_train, X_test, y_train, y_test, title):
	nn_params = {
					"activation": "logistic",
					"alpha": 1e-06, 
					"hidden_layer_sizes": (64, 32, 16), 
					"learning_rate": "adaptive", 
					"learning_rate_init": 0.0001, 
					"max_iter": 1000,
					"solver": "adam",
					"random_state": 123565432
				}

	model = MLPClassifier(**nn_params)

	train_start = time()
	model.fit(X_train, y_train)
	train_time = time() - train_start

	train_acc = model.score(X_train, y_train)

	test_start = time()
	test_acc = model.score(X_test, y_test)
	test_time = time() - test_start

	print ("=================")
	print ("For %s:" % title)
	print ("Training Accuracy: %.4f" % train_acc)
	print ("Testing Accuracy: %.4f" % test_acc)
	print ("Training Time: %.4f" % train_time)
	print ("Testing: %.4f" % test_time)

def prepare_part5_data():
	DR = ["PCA", "ICA", "RP", "MI"]
	kmeans_param = [20, 8, 20, 2]
	gmm_param = [30, 5, 30, 2]
	for dr, k_kmeans, k_gmm in zip(DR, kmeans_param, gmm_param):
		file_path = "data/MNIST/%s.csv" % dr
		X, y, _, _ = load_data(file_path, is_shuffle=True, is_split=False)
		
		kmean = KMeans(n_clusters=k_kmeans, random_state=10)
		cluster_labels = kmean.fit_predict(X)
		data = np.hstack((np.array([cluster_labels]).T, np.array([y]).T))
		path = create_path('data', "MNIST", filename='%s+KMeans.csv' % dr)
		np.savetxt(path, data, delimiter=",")

		gmm = GaussianMixture(n_components=k_gmm, covariance_type='full', random_state=10)
		cluster_labels = gmm.fit_predict(X)
		data = np.hstack((np.array([cluster_labels]).T, np.array([y]).T))
		path = create_path('data', "MNIST", filename='%s+GMM.csv' % dr)
		np.savetxt(path, data, delimiter=",")

if __name__ == "__main__":
	if len(sys.argv) != 2 or sys.argv[1] not in ["part4", "part5"]:
		print ("Usage: python NN.py [part4|part5]")
		exit(1)

	if sys.argv[1] == "part4":
		file_path = "data/MNIST/MNIST_4_9_size-1000.csv"
		X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		run_NN(X_train, X_test, y_train, y_test, "Non reduced")

		file_path = "data/MNIST/PCA.csv"
		X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		run_NN(X_train, X_test, y_train, y_test, "PCA")

		file_path = "data/MNIST/ICA.csv"
		X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		run_NN(X_train * 10000, X_test * 10000, y_train, y_test, "ICA")

		file_path = "data/MNIST/RP.csv"
		X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		run_NN(X_train, X_test, y_train, y_test, "RP")

		file_path = "data/MNIST/MI.csv"
		X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		run_NN(X_train, X_test, y_train, y_test, "MI")
	else:
		#prepare_part5_data()
		i = 0
		scaling = [5, 10, 15, 20, 5, 5, 10, 10]
		for dr in ["PCA", "ICA", "RP", "MI"]:
			for cluster_alg in ["KMeans", "GMM"]:
				file_path = "data/MNIST/%s+%s.csv" % (dr, cluster_alg)
				X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
				run_NN(X_train * scaling[i], X_test * scaling[i], y_train, y_test, "%s + %s" % (dr, cluster_alg))

				cluster_labels = np.vstack((X_train, X_test)).T[0]
				true_labels = np.hstack((y_train, y_test))
				v_measure(cluster_labels, true_labels)
				i += 1



