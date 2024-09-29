import numpy as np
from collections import defaultdict
from typing import Dict,List

from knee_locator import KneeLocator
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram


class DataCorrelationAnalysis():
    def __init__(self,X):
        self.X = X
   
    def calculate_cov(self):
        self.cov = np.cov(self.X, rowvar=False)
   
    def draw_heatmap(self):
        sns.heatmap(self.cov, annot=True,
                    cmap='coolwarm')
        plt.show()


class DataPreprocessTypeError(Exception):
    def __init__(self,message,errors):
        super().__init__(message)
        self.errors = errors
        

class DataReduction():
    def __init__(self,method):
        if method == 'PCA':
            self.reduction_func = self.reduction_PCA
        elif method == 'TSNE':
            self.reduction_func = self.reduction_TSNE
        elif method == 'TSVD':
            self.reduction_func = self.reduction_TSVD
        else:
            error_message = 'please select reduction type from PCA, TSNE or TSVD'
            raise DataPreprocessTypeError(error_message)
   
    def reduction_PCA(self,data,n_components):
        from sklearn.decomposition import PCA
        self.reduction_model = PCA(n_components=n_components)
        return self.reduction_model.fit_transform(data)
   
    def reduction_TSNE(self,data,n_components):
        from sklearn.manifold import TSNE
        self.reduction_model = TSNE(n_components=n_components)
        return self.reduction_model.fit_transform(data)
   
    def reduction_TSVD(self,data,n_components):
        from sklearn.decomposition import TruncatedSVD
        self.reduction_model = TruncatedSVD(n_components=n_components)
        return self.reduction_model.fit_transform(data)


class Scale():
    def __init__(self,method):
        if method == 'minmax':
            self.scale_func = self.minmax_scaling
        elif method == 'standard':
            self.scale_func = self.standardization
        else:
            error_message = 'please select Norm or scaling type from minmax, standard'
            raise DataPreprocessTypeError(error_message)
       
    def minmax_scaling(self,data):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(data)
   
    def standardization(self,data):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(data)
    

class DataPreprocessing():
   
    def __init__(self,raw_data):
        self.data = raw_data
   
    def analyze_cov(self):
        analyzer = DataCorrelationAnalysis(self.data)
        analyzer.calculate_cov()
        analyzer.draw_heatmap()
   
    def scaling(self,method,norm='l2'):
        self.scale_obj = Scale(method)
        self.data = self.scale_obj.scale_func(self.data)
       
    def reduce_dimension(self,method,n_components):
        self.reduction_obj = DataReduction(method)
        self.data = self.reduction_obj.reduction_func(self.data,n_components)
       
   
def FindKnee(X,Y):
    kneedle = KneeLocator(X, Y, curve="convex", direction="decreasing")
    elbow_point = kneedle.elbow
    return elbow_point
   
def FindKneeForModel(model_func):
    def wrapper(self, X, expected_cluster_magnitude:int):
        '''estimate 50, magnitude = 2'''
        magnitude = expected_cluster_magnitude
        elbow_point = int(10**magnitude/2)
        while magnitude >= 1:
            lower_mangnitude = int(10**(magnitude-1))
            left = elbow_point-4*lower_mangnitude
            right = elbow_point+5*lower_mangnitude
            k_range,evaluation = range(left,right,lower_mangnitude),[]
            for k in k_range:
                evaluation.append(model_func(self, X, k))
                magnitude -= 1
            elbow_point = FindKnee(k_range,evaluation)
        return elbow_point
    return wrapper


class ClusterModel():
    

    def combine_df(self,df,label):
        self.df = df
        self.df['label'] = label
   
    def save_combined_df(self,path):
        self.df.to_csv(path)
   
    def reduce_D(self,data,n_components):
        reduction_model = PCA(n_components=n_components)
        return reduction_model.fit_transform(data)
   
    def reduce_to_3D_data(self,data):
        return self.reduce_D(data,3)
   
    def reduce_to_2D_data(self,data):
        return self.reduce_D(data,2)
   
    def display_3D_clusters(self,X_3D,labels,core_samples_mask=False,figsize=(8, 6)):
        if type(core_samples_mask)==bool:
            core_samples_mask = np.ones_like(labels, dtype=bool)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xyz = X_3D[class_member_mask & core_samples_mask]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], marker='o')
            xyz = X_3D[class_member_mask & ~core_samples_mask]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], marker='x')
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
       
    def display_2D_clusters(self,X_2D,labels,core_samples_mask=False,figsize=(8, 6)):
        if type(core_samples_mask)==bool:
            core_samples_mask = np.ones_like(labels, dtype=bool)
        plt.figure(figsize=figsize)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X_2D[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
            xy = X_2D[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()


class ClusterKmeans(ClusterModel):
       
    def Kmeans_model(self,X,n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        return kmeans.labels_
   
    @FindKneeForModel
    def Kmeans_Knee_optimize(self,X,k:int):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        return kmeans.inertia_
   
    def Kmeans_grid_optimize(self,X,k_range=range(1,10)):
        best_score = -1
        best_params = {}
        for k in k_range:
            km = KMeans(n_clusters=k)
            labels = km.fit(X).labels_
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = k
        return best_params


class ClusterHierachical(ClusterModel):
       
    def Hierachical(self,X,best_params):
        H_metric = best_params['metric']
        H_method = best_params['linkage']
        model = AgglomerativeClustering(metric=H_metric, linkage=H_method)
        labels = model.fit_predict(X)
        return labels
   
    def Hierachical_grid_optimize(self,X):
        distance_metrics = ['euclidean', 'manhattan', 'cosine']
        linkage_methods = ['single', 'complete', 'average', 'ward']
        best_score = -1
        best_params = {}
        for metric in distance_metrics:
            for method in linkage_methods:
                if method == 'ward' and metric != 'euclidean':
                    continue
                model = AgglomerativeClustering(metric=metric, linkage=method)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels, metric=metric)
                if score > best_score:
                    best_score = score
                    best_params = {'metric': metric, 'linkage': method}
        return best_params
   
    def Hierachical_display(self,X,best_params,figsize=(10, 7)):
        H_metric = best_params['metric']
        H_method = best_params['linkage']
        Z = linkage(X, method=H_method, metric=H_metric)
        plt.figure(figsize=figsize)
        dendrogram(Z)
        plt.show()


class ClusterDBSCAN(ClusterModel):
       
    def DBSCAN_(self,X,best_params):
        eps = best_params['eps']
        min_samples = best_params['min_samples']
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        return labels, core_samples_mask
       
    def DBSCAN_optimize(self,X,
                        eps_values= np.arange(0.1, 1.0, 0.1),
                        min_samples_values = range(1, 10)):
        best_score = -1
        best_params = {}
        for eps in eps_values:
            for min_samples in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(X)
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        return best_params


def bic_score(estimator, X):
    return -estimator.bic(X)


class ClusterGMM(ClusterModel):
   
    def GMM_model(self,X,best_params):
        n_components = best_params['n_components']
        covariance_type = best_params['covariance_type']
        gmm = GaussianMixture(n_components = n_components,
                              covariance_type = covariance_type)
        gmm.fit(X)
        labels = gmm.predict(X)
        return labels
   
    def GMM_optimize(self,X,n_components=range(1,10)):
        param_dist = {
            'n_components': n_components,
            'covariance_type': ['full', 'tied', 'diag', 'spherical']
        }
        random_search = RandomizedSearchCV(estimator=GaussianMixture(),
                                        param_distributions=param_dist,
                                        n_iter=20, scoring=bic_score, cv=5)
        random_search.fit(X)
        return random_search.best_params_


def collect_label_dict(label):
    label_dict = defaultdict(list)
    for i,l in enumerate(label):
        label_dict[l].append(i)
    return label_dict


def get_list_above_threshold(label_dict:Dict,threshold:int):
    above_label_list = []
    for k,V in label_dict.items():
        if len(V) > threshold:
            above_label_list.append(V)
    return above_label_list    


def get_new_label(label,X,above_label_list,cluster_func):
    max_label = max(label)+1
    for above_label in above_label_list:
        this_X = X[above_label,:]
        this_label= cluster_func(this_X) + max_label
        label[above_label] = this_label
        max_label = max(label)+1
    return sort_label(label)


def sort_label(label):
    label_unique = sorted(list(set(label)))
    label_unique_dict = {l:i for i,l in enumerate(label_unique)}
    return np.array([label_unique_dict[l] for l in label])


def screen_under_threshold(cluster_func):
    def wrapper(X,max_cluster_threshold):
        label = np.array([-1 for i in range(X.shape[0])])
        label_list = collect_label_dict(label)
        above_label_list = get_list_above_threshold(label_list,max_cluster_threshold)
        while above_label_list:
            label = get_new_label(label,X,above_label_list,cluster_func)
            label_list = collect_label_dict(label)
            above_label_list = get_list_above_threshold(label_list,max_cluster_threshold)
        return label
    return wrapper

@screen_under_threshold
def kmeans_model_cluster(X):
    kmeans_model = ClusterKmeans()
    n_clusters = kmeans_model.Kmeans_grid_optimize(X,range(1,10))
    return kmeans_model.Kmeans_model(X,n_clusters)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, n_features= 6, centers=5, random_state=42)
   
    DataPrecessor = DataPreprocessing(X)
    DataPrecessor.scaling('standard')
    DataPrecessor.reduce_dimension('PCA',4)
    DataPrecessor.scaling('standard')
    #DataPrecessor.analyze_cov()
    X = DataPrecessor.data
   
    '''
    #kmeans model
    kmeans_model = ClusterKmeans()
    n_clusters = kmeans_model.Kmeans_Knee_optimize(X,1)
    n_clusters = kmeans_model.Kmeans_grid_optimize(X,range(1,10))
    labels = kmeans_model.Kmeans_model(X,n_clusters)
    X_3D  = kmeans_model.reduce_to_3D_data(X)
    kmeans_model.display_3D_clusters(X_3D,labels)
    X_2D  = kmeans_model.reduce_to_2D_data(X)
    kmeans_model.display_2D_clusters(X_2D,labels)
   
    #Hierachical model
    hierachical_model = ClusterHierachical()
    best_params = hierachical_model.Hierachical_grid_optimize(X)
    labels = hierachical_model.Hierachical(X,best_params)
    hierachical_model.Hierachical_display(X,best_params)
   
    #DBSCAN model
    DBSCAN_model = ClusterDBSCAN()
    best_params = DBSCAN_model.DBSCAN_optimize(X,
                                                eps_values= np.arange(0.1, 1.0, 0.1),
                                                min_samples_values = range(1, 10))
    labels,core_samples_mask = DBSCAN_model.DBSCAN_(X,best_params)
    X_3D  = DBSCAN_model.reduce_to_3D_data(X)
    DBSCAN_model.display_3D_clusters(X_3D,labels,core_samples_mask)
    X_2D  = DBSCAN_model.reduce_to_2D_data(X)
    DBSCAN_model.display_2D_clusters(X_2D,labels,core_samples_mask)
   
   
    #GMM model
    GMM = ClusterGMM()
    best_params = GMM.GMM_optimize(X,n_components=range(1,10))
    labels = GMM.GMM_model(X,best_params)
    X_3D  = GMM.reduce_to_3D_data(X)
    GMM.display_3D_clusters(X_3D,labels)
    X_2D  = GMM.reduce_to_2D_data(X)
    GMM.display_2D_clusters(X_2D,labels)
    '''
   
   

