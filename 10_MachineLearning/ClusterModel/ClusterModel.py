import numpy as np

from knee_locator import KneeLocator
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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
    
    def Hierachical(self):
        pass
    
    def Hierachical_optimize(self,):
        pass
    
    def DNSCAN(self):
        pass
    
    def DBSCAN_optimize(self):
        pass