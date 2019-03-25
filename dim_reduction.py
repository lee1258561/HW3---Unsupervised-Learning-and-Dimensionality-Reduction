import sys
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import mutual_info_classif, SelectPercentile, SelectKBest
from sklearn.cluster import KMeans
from scipy.stats import kurtosis

import numpy as np
import matplotlib.pyplot as plt

from utils import *

def PCA_experiment(data_path, task):
	X, y, _, _ = load_data(data_path, is_shuffle=True, is_split=False)
	pca = PCA(random_state=10)
	pca.fit(X)

	fig_path = create_path('fig', task, 'PCA', filename="eigenvalue")
	plot_and_save(list(range(len(pca.explained_variance_))), 
				  [pca.explained_variance_], [], 
				  "%s - PCA Eigenvalue" % task, "n_component", "eigenvalue", 
				  fig_path=fig_path, format='png')

def ICA_experiment(data_path, task):
	cols, rows = 3, 2
	X, y, _, _ = load_data(data_path, is_shuffle=True, is_split=False)
	sizes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
	for size in sizes:
		n_components = int(np.ceil(X.shape[1] * size))
		ica = FastICA(n_components=n_components, random_state=10)
		X_transformed = ica.fit_transform(X)
		X_kurtosis = kurtosis(X_transformed, axis=0)

		print (X_kurtosis - 3)
		fig_path = create_path('fig', task, 'ICA', filename="%d_kurtosis" % n_components)
		plot_and_save(list(range(X_kurtosis.shape[0])), 
					  [X_kurtosis], [], 
					  "%s - ICA kurtosis" % task, "n_component", "kurtosis", 
					  fig_path=fig_path, format='png')

		if task == "MNIST":
			fig_path = create_path('fig', task, 'ICA', filename="%d_componentVis.png" % n_components)
			plot_gallery('ICA MNIST', ica.components_[:(cols * rows)], fig_path, n_col=cols, n_row=rows, shape=(28, 28))

def RP_experiment(data_path, task, k, trials=10):
	X, y, _, _ = load_data(data_path, is_shuffle=True, is_split=False)
	sizes = list(map(lambda x: x / 30, list(range(1, 31))))

	sse_score, reconstruction_error, n_components_list = [], [], []
	for size in sizes:
		# avg_sse = 0
		# min_sse = float("inf")
		min_re_error = float("inf")
		n_components = int(np.ceil(X.shape[1] * size))
		n_components_list.append(n_components)
		for i in range(trials):
			rp = GaussianRandomProjection(n_components=n_components)
			X_transformed = rp.fit_transform(X)
			X_reconstructed = np.dot(X_transformed, rp.components_)

			error = np.mean((X - X_reconstructed) ** 2)
			min_re_error = min(error, min_re_error)

			# clusterer = KMeans(n_clusters=k)
			# cluster_labels = clusterer.fit_predict(X_transformed)
			# avg_sse += clusterer.inertia_
			# min_sse = min(clusterer.inertia_, min_sse)

		#sse_score.append(avg_sse / trials)
		# sse_score.append(min_sse)
		print ("Reconstruction Error for n_components = %d: %.6f" % (n_components, min_re_error))
		reconstruction_error.append(min_re_error)

	fig_path = create_path('fig', task, 'RP', filename="reconstruction_error")
	plot_and_save(n_components_list, 
				  [reconstruction_error], [], 
				  "%s - RP Reconstruction error" % task, "n_component", "re error", 
				  fig_path=fig_path, format='png')

def MI_experiment(data_path, task):
	X, y, _, _ = load_data(data_path, is_shuffle=True, is_split=False)
	mi = mutual_info_classif(X, y)
	fig_path = create_path('fig', task, 'MI', filename="mutual_info")
	plot_and_save(list(range(mi.shape[0])), 
				  [mi], [], 
				  "%s - Mutual Information" % task, "feature dimension", "mi", 
				  fig_path=fig_path, format='png')



def reconstruction_error(data_path, task):
	X, y, _, _ = load_data(data_path, is_shuffle=True, is_split=False)
	sizes = list(map(lambda x: x / 30, list(range(1, 31))))
	for size in sizes:
		n_components = int(np.ceil(X.shape[1] * size))


def dump_data(data_path, task, reduce_sizes, trials=10):
	X, y, _, _ = load_data(data_path, is_shuffle=True, is_split=False)
	pca_components = reduce_sizes[0]
	pca = PCA(n_components=pca_components, random_state=10)
	X_PCA = pca.fit_transform(X)
	X_reconstructed = pca.inverse_transform(X_PCA)
	print("Reconstruction Error for PCA: %.6f" % np.mean((X - X_reconstructed) ** 2))

	data = np.hstack((X_PCA, np.array([y]).T))
	PCA_path = create_path('data', task, filename='PCA.csv')
	np.savetxt(PCA_path, data, delimiter=",")

	ica_components = reduce_sizes[1]
	ica = FastICA(n_components=ica_components, random_state=10)
	X_ICA = ica.fit_transform(X)
	X_reconstructed = ica.inverse_transform(X_ICA)
	print("Reconstruction Error for ICA: %.6f" % np.mean((X - X_reconstructed) ** 2))

	data = np.hstack((X_ICA, np.array([y]).T))
	ICA_path = create_path('data', task, filename='ICA.csv')
	np.savetxt(ICA_path, data, delimiter=",")

	rp_components = reduce_sizes[2]
	re_list = []
	min_re_error = float("inf")
	X_RP = None
	for i in range(trials):
		rp = GaussianRandomProjection(n_components=rp_components)
		rp.fit(X)
		X_transformed = rp.transform(X)
		c_square = np.dot(rp.components_.T, rp.components_)
		X_reconstructed = np.dot(X_transformed, rp.components_)

		error = np.mean((X - X_reconstructed) ** 2)
		if error < min_re_error:
			min_re_error = error
			X_RP = X_transformed

		re_list.append(error)

	print (np.mean(re_list))
	print (np.std(re_list))
	print("Reconstruction Error for RP: %.6f" % min_re_error)

	data = np.hstack((X_RP, np.array([y]).T))
	RP_path = create_path('data', task, filename='RP.csv')
	np.savetxt(RP_path, data, delimiter=",")

	mi_components = reduce_sizes[3]
	X_MI = SelectKBest(mutual_info_classif, k=mi_components).fit_transform(X, y)
	data = np.hstack((X_MI, np.array([y]).T))
	MI_path = create_path('data', task, filename='MI.csv')
	np.savetxt(MI_path, data, delimiter=",")

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print ("Usage: python dim_reduction.py [creditCard|MNIST] [dump|exp]")
		exit(1)

	if sys.argv[1] == "MNIST":
		data_path = "data/MNIST/MNIST_4_9_size-1000.csv"
	elif sys.argv[1] == "creditCard":
		data_path = "data/creditCard/size-5000_porp-0.1.csv"
	else:
		print ("Usage: python kmean.py [creditCard|MNIST]")
		exit(1)

	if sys.argv[2] == "exp":
		if sys.argv[1] == "MNIST": k = 20
		elif sys.argv[1] == "creditCard": k = 8

		PCA_experiment(data_path, sys.argv[1])
		ICA_experiment(data_path, sys.argv[1])
		RP_experiment(data_path, sys.argv[1], k)
		MI_experiment(data_path, sys.argv[1])
	elif sys.argv[2] == "dump":
		if sys.argv[1] == "MNIST":
			size = [115, 79, 150, 79]
		elif sys.argv[1] == "creditCard":
			size = [2, 3, 6, 3]
		dump_data(data_path, sys.argv[1], size)
	else:
		print ("Usage: python dim_reduction.py [creditCard|MNIST] [dump|exp]")
		exit(1)
