import numpy as np
from knee_locator import KneeLocator
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


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
    
class NormScale():
    def __init__(self,method,norm ='l1'):
        self.norm = norm
        if method == 'minmax':
            self.scale_norm_func = self.minmax_scaling
        elif method == 'standard':
            self.scale_norm_func = self.standardization
        elif method == 'norm':
            self.scale_norm_func = self.normalizarion
        else:
            error_message = 'please select Norm or scaling type from minmax, standard or norm'
            raise DataPreprocessTypeError(error_message)
    def minmax_scaling(self,data):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(data)
    
    def standardization(self,data):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(data)
    
    def normalizarion(self,data):
        from sklearn.preprocessing import normalize
        return normalize(data, norm=self.norm)

def FindKnee(X,Y):
    kneedle = KneeLocator(X, Y, curve="convex", direction="decreasing")
    elbow_point = kneedle.elbow
    return elbow_point
    
def FindKneeForModel(mode_func):
    def wrapper(self, expected_cluster_magnitude:int):
        '''estimate 50, magnitude = 2'''
        magnitude = expected_cluster_magnitude
        elbow_point = 10**magnitude/2
        while magnitude >= 1:
            lower_mangnitude = 10**(magnitude-1)
            left = elbow_point-4*lower_mangnitude
            right = elbow_point+5*lower_mangnitude
            k_range,evaluation = range(left,right,lower_mangnitude),[]
            for k in k_range:
                evaluation.append(mode_func(self,k))
                magnitude -= 1
                elbow_point = FindKnee(k_range,evaluation)
            return elbow_point
        return wrapper


class ClusterModel():
    def __init__(self,raw_data):
        self.data = raw_data
    def reduce_dimension(self,method,n_components):
        self.reduction_obj = DataReduction(method)
        self.data = self.reduction_obj.reduction_func(self.data,n_components)
    def norm_scaling(self,method,norm='l2'):
        self.norm_scale_obj = NormScale(method,norm)
        self.data = self.norm_scale_obj.scale_norm_func(self.data)
    
    @FindKneeForModel
    def Kmeans_optimize(self,k:int):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.data)
        return kmeans.inertia_
    
    def Hierachical(self,X):
        best_params = self.Hierachical_optimize(X)
        self.H_metric = best_params['metric']
        self.H_method = best_params['linkage']
        model = AgglomerativeClustering(affinity=self.H_metric, linkage=self.H_method)
        labels = model.fit_predict(X)
        return labels
    
    def Hierachical_optimize(self,X):
        distance_metrics = ['euclidean', 'manhattan', 'cosine']
        linkage_methods = ['single', 'complete', 'average', 'ward']
        best_score = -1
        best_params = {}
        for metric in distance_metrics:
            for method in linkage_methods:
                if method == 'ward' and metric != 'euclidean':
                    continue
                model = AgglomerativeClustering(affinity=metric, linkage=method)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels, metric=metric)
                if score > best_score:
                    best_score = score
                    best_params = {'metric': metric, 'linkage': method}
        return best_params
    
    def Hierachical_display(self,X):
        Z = linkage(X, method=self.H_method, metric=self.H_metric)
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.show()
    
    def DBSCAN(self,X):
        db = DBSCAN(eps=0.3, min_samples=5)
        labels = db.fit_predict(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    def DBSCAN_optimize(self,X,
                        eps_values= np.arange(0.1, 1.0, 0.1),
                        min_samples_values = range(3, 10)):
        best_score = -1
        best_params = {}
        for eps in eps_values:
            for min_samples in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(X)
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    print(f'EPS: {eps}, Min_samples: {min_samples}, Silhouette Score: {score:.4f}')
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        return best_params
    
    def DBSCAN_display(self,X,labels,core_samples_mask,n_clusters_):
        plt.figure(figsize=(8, 6))
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    
    def GMM_model(self,X):
        gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
    
    def GMM_optimize(self,X,n_components=range(1, 10)):
        param_dist = {
            'n_components': n_components,
            'covariance_type': ['full', 'tied', 'diag', 'spherical']
        }
        random_search = RandomizedSearchCV(estimator=GaussianMixture(random_state=42),
                                        param_distributions=param_dist,
                                        n_iter=20, scoring='neg_bic', cv=5, random_state=42)
        random_search.fit(X)
        return random_search.best_params_