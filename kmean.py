import sys
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

from utils import *

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print ("Usage: python kmean.py [creditCard|MNIST] [nonReduced|PCA|ICA|RP|MI]")
        exit(1)

    file_path = ""
    if sys.argv[1] == "creditCard":
        if sys.argv[2] == "nonReduced":
            file_path = "data/creditCard/size-5000_porp-0.1.csv"
        elif sys.argv[2] == "PCA":
            file_path = "data/creditCard/PCA.csv"
        elif sys.argv[2] == "ICA":
            file_path = "data/creditCard/ICA.csv"
        elif sys.argv[2] == "RP":
            file_path = "data/creditCard/RP.csv"
        elif sys.argv[2] ==  "MI":
            file_path = "data/creditCard/MI.csv"
    elif sys.argv[1] == "MNIST":
        if sys.argv[2] == "nonReduced":
            file_path = "data/MNIST/MNIST_4_9_size-1000.csv"
        elif sys.argv[2] == "PCA":
            file_path = "data/MNIST/PCA.csv"
        elif sys.argv[2] == "ICA":
            file_path = "data/MNIST/ICA.csv"
        elif sys.argv[2] == "RP":
            file_path = "data/MNIST/RP.csv"
        elif sys.argv[2] ==  "MI":
            file_path = "data/MNIST/MI.csv"

    X, y, _, _ = load_data(file_path, is_shuffle=True, is_split=False)
    pca_full = PCA(random_state=10)
    pca_full.fit(X)
    print("Precentage of covarence preserved: %0.03f" % np.sum(pca_full.explained_variance_ratio_[:2]))
    pca = PCA(n_components=2, random_state=10)
    pca.fit(X) 
    X_vis = pca.transform(X)
    print (X_vis.shape, X.shape)

    range_n_clusters = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50]
    sse_score, h_score, c_score, v_score = [], [], [], []
    ari_score, ami_score, nmi_score, fms_score, sil_score, chi_score, dbi_score = [], [], [], [], [], [], []

    for n_clusters in range_n_clusters:
        print ("============")
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        sse_score.append(clusterer.inertia_)


        # figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename=("%d.png" % n_clusters))
        # silhouette_analysis(X, cluster_labels, n_clusters, figname)

        centers = pca.transform(clusterer.cluster_centers_)
        figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename=("%d_vis.png" % n_clusters))
        visualize_cluster(X_vis, cluster_labels, n_clusters, centers, figname)

        ari = metrics.adjusted_rand_score(y, cluster_labels)
        ami = metrics.adjusted_mutual_info_score(y, cluster_labels)
        nmi = metrics.normalized_mutual_info_score(y, cluster_labels)
        fms = metrics.fowlkes_mallows_score(y, cluster_labels) 
        sil = metrics.silhouette_score(X, cluster_labels, metric='euclidean')
        chi = metrics.calinski_harabaz_score(X, cluster_labels)
        dbi = metrics.davies_bouldin_score(X, cluster_labels)

        print ("Adjusted Rand index: %.6f" % ari)
        print ("Adjusted Mutual Information: %.6f" % ami)
        print ("Normalized Mutual Information: %.6f" % nmi)
        print ("Fowlkes-Mallows score: %.6f" % fms)
        print ("Silhouette Coefficient: %.6f" % sil)
        print ("Calinski-Harabaz Index: %.6f" % chi)
        print ("Davies-Bouldin Index: %.6f" % dbi)

        ari_score.append(ari)
        ami_score.append(ami)
        nmi_score.append(nmi)
        fms_score.append(fms)
        sil_score.append(sil)
        chi_score.append(chi)
        dbi_score.append(dbi)

        print ("SSE score: %.6f" % clusterer.inertia_)

        print ("V Measure for n_clusters = %d: " % n_clusters)
        h, c, v = v_measure(cluster_labels, y)
        h_score.append(h)
        c_score.append(c)
        v_score.append(v)

        figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_ari") 
    plot_and_save(range_n_clusters, 
                  [ari_score], 
                  [], 
                  "KMeans Adjusted Rand index", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_mi") 
    plot_and_save(range_n_clusters, 
                  [ami_score, nmi_score], 
                  ["Adjusted Mutual Information", "Normalized Mutual Information"], 
                  "KMeans Mutual Information", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_fms") 
    plot_and_save(range_n_clusters, 
                  [fms_score], 
                  [], 
                  "KMeans Fowlkes-Mallows score", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_sil") 
    plot_and_save(range_n_clusters, 
                  [sil_score], 
                  [], 
                  "KMeans Silhouette Coefficient", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_chi") 
    plot_and_save(range_n_clusters, 
                  [chi_score], 
                  [], 
                  "KMeans Calinski-Harabaz Index", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_dbi") 
    plot_and_save(range_n_clusters, 
                  [dbi_score], 
                  [], 
                  "KMeans Davies-Bouldin Index", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_score") 
    plot_and_save(range_n_clusters, 
                  [sse_score], 
                  ["SSE"], 
                  "KMeans Score", "n_clusters", "score", 
                  fig_path=figname, format='png')

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="kmeans_v_measure") 
    plot_and_save(range_n_clusters, 
                  [h_score, c_score, v_score], 
                  ["Homogeneity", "Completeness", "V Measure"], 
                  "KMeans V Measure", "n_clusters", "score", 
                  fig_path=figname, format='png') 

    figname = create_path("fig", sys.argv[1], "KMeans", sys.argv[2], filename="true.png")
    visualize_cluster(X_vis, y, 2, [], figname)
        

