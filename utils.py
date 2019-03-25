from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn import datasets
from random import shuffle

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings
warnings.simplefilter("ignore")

def v_measure(cluster_labels, true_labels):
    h_score = homogeneity_score(true_labels, cluster_labels)
    c_score = completeness_score(true_labels, cluster_labels)
    v_score = v_measure_score(true_labels, cluster_labels)

    print("Homogeneity Score: %.6f" % h_score)
    print("Completeness Score: %.6f" % c_score)
    print("V Measure Score: %.6f" % v_score)
    return h_score, c_score, v_score

def silhouette_analysis(X, cluster_labels, n_clusters, figname):
    plt.xlim([-0.1, 1])
    plt.ylim([0, len(X) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(figname, format='png')
    plt.clf()

def visualize_cluster(X, cluster_labels, n_clusters, centers, figname):
    if X.shape[1] < 2:
        print ("Invalid shape for X: ", X.shape)
        return

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Draw white circles at cluster centers
    if len(centers) == n_clusters:
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

    plt.title("The visualization of the clustered data.")
    plt.xlabel("Feature space for the 1st feature")
    plt.ylabel("Feature space for the 2nd feature")
    plt.savefig(figname, format='png')
    plt.clf()

def plot_gallery(title, images, figname, n_col=3, n_row=2, shape=(28, 28), cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(shape), cmap=cmap,
                   interpolation='nearest',
                vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    #plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.savefig(figname, format='png')
    plt.clf()
    #plt.subplots_adjust()

def create_path(*arg, filename=None):
    path = os.getcwd()
    for directory in arg:
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            print('%s doesn\'t exist, creating...' % path)
            os.mkdir(path)

    if filename:
        path = os.path.join(path, filename)
    return path

def load_data(data_path, split_prop=0.2, is_shuffle=False, is_split=True):
    pos_X, neg_X = [], []
    with open(data_path, 'r') as f:
        for line in f:
            instance = list(map(float, line.strip().split(',')))
            if instance[-1] == 1.0:
                pos_X.append(instance[:-1])
            else:
                neg_X.append(instance[:-1])

    if not is_split:
        X, y = np.array(pos_X + neg_X), np.array([1] * len(pos_X) + [0] * len(neg_X))
        if is_shuffle:
            indices = list(range(X.shape[0]))
            shuffle(indices)
            X, y = X[indices], y[indices]
            return X, y, [], []

    pos_test_size, neg_test_size = int(split_prop * len(pos_X)), int(split_prop * len(neg_X))
    pos_train_size, neg_train_size = len(pos_X) - pos_test_size, len(neg_X) - neg_test_size
    
    X_test, y_test = pos_X[:pos_test_size] + neg_X[:neg_test_size], [1] * pos_test_size + [0] * neg_test_size
    X_train, y_train = pos_X[pos_test_size:] + neg_X[neg_test_size:], [1] * pos_train_size + [0] * neg_train_size

    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "Dimention of X and y must be the same."

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    if is_shuffle:
        train_indices = list(range(X_train.shape[0]))
        shuffle(train_indices)
        test_indices = list(range(X_test.shape[0]))
        shuffle(test_indices)
        X_train, X_test, y_train, y_test = X_train[train_indices], X_test[test_indices], y_train[train_indices], y_test[test_indices]

    return X_train, X_test, y_train, y_test

def dump_data():
    #Need implement
    pass

def analyze_data(data_path, threshold=50):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            instance = list(map(float, line.strip().split(',')))
            data.append(instance)

    count = [0] * len(data[0])
    for instance in data:
        for i in range(len(instance)):
            if instance[i] != 0.0:
                count[i] += 1

    total = 0
    for c in count:
        if c >= threshold:
            total += 1


    return count, total

def plot_learning_curve(train_scores_mean,
                        train_scores_std,
                        val_scores_mean,
                        val_scores_std,
                        train_sizes,
                        ylim=None,
                        title='test',
                        fig_path='fig',
                        format='png'):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid(True, linestyle = "-.", color = '0.3')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(fig_path + '/' + title + '.' + format, format=format)
    plt.clf()

def plot_and_save(x, ys, labels, title, x_axis, y_axis, axis_range='auto', ylim=None, fig_path='fig', format='png'):
    if axis_range is None:
        plt.axis([x[0], x[-1], 0, 1])
    elif type(axis_range) == type(list()):
        plt.axis(axis_range)
    elif axis_range == 'auto':
        pass

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    lines = []
    for y in ys:
        l, = plt.plot(x, y)
        lines.append(l)
    if len(labels) == len(ys):
        plt.legend(lines, labels, loc="best")
    plt.grid(True, linestyle = "-.", color = '0.3')

    plt.savefig(fig_path + '.' + format, format=format)
    plt.clf()

def print_score(scores, scoring, train=False):
    if type(scoring) != type([]):
        if train:
            print("Train: %0.2f (+/- %0.2f)" % (np.mean(scores['train_score']), np.std(scores['train_score']) * 2))

        print("Cross validation: %0.2f (+/- %0.2f)" % (np.mean(scores['test_score']), np.std(scores['test_score']) * 2))
        return

    for s_method in scoring:
        if train:
            print("Train: %0.2f (+/- %0.2f)" % (np.mean(scores['train_' + s_method]), np.std(scores['train_' + s_method]) * 2))

        print("Cross validation: %0.2f (+/- %0.2f)" % (np.mean(scores['test_' + s_method]), np.std(scores['test_' + s_method]) * 2))


    