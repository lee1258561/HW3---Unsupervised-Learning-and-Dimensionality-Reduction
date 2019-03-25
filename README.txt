0. git clone https://github.com/lee1258561/HW3---Unsupervised-Learning-and-Dimensionality-Reduction.git
	The data and model parameters in this repository is sufficient to reproduce the result.
1. Setup
	using python 3.7.2
	run:
		pip install sklearn matplotlib

3. Usage (run under root dir):
	python python kmean.py [creditCard|MNIST] [nonReduced|PCA|ICA|RP|MI]
		Run KMeans on the specific dataset with or without the specific dimensionality reduction algorithm.
		
	python gaussian_mixture.py [creditCard|MNIST] [nonReduced|PCA|ICA|RP|MI] 
		Run Expectation Maximization on the specific dataset with or without the specific dimensionality reduction algorithm.

	python python dim_reduction.py [creditCard|MNIST] [dump|exp]
		If exp is specified, run experiment for the four dimensionality reduction algorithms on the specific dataset. If dump is specified, conduct four dimensionality reduction algorithms on the specific dataset with the best hyperparameter and dump the data.

	python NN.py [part4|part5]
		Run part4 or part5 experiment with respect to the neural network.


