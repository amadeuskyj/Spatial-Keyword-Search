#from Cluster import Cluster
from scipy.spatial import distance
#from rtree import index
import pandas as pd
import scipy.spatial as spatial

class DBSCAN:
    """
    DBSCAN Class
    """
    def __init__(self, eps, minPts):
        """
        eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

        minPts:
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        """
        self.eps = eps
        self.minPts = minPts
        self.num_clusters = 0

    def predict(self, data):
        """
        Data is Pandas dataframe of 2 columns (latitude, longitude)
        """
        #self.rtree = index.Index()
        self.data = data
        self.label = [None] * len(data)
        
        self.preprocess_data()
        # for i in range(len(data)):
        #     self.rtree.insert(i, (self.data[i][0], self.data[i][1], self.data[i][0], self.data[i][1]))

        self.kdtree = spatial.cKDTree(self.data)

        for i in range(len(data)):
            #print("Now at point " + str(i))
            if self.label[i] == None:
                neighbours = self.region_query(i)
                if len(neighbours) < self.minPts:
                    self.label[i] = -1
                else:
                    self.expand_cluster(self.num_clusters, i, neighbours)
                    self.num_clusters += 1
        return self.label

    def expand_cluster(self, cluster_id, point, neighbours):
        self.label[point] = cluster_id
        for pt in neighbours:
            if self.label[pt] == -1:
                self.label[pt] = cluster_id
            elif self.label[pt] == None:
                self.label[pt] = cluster_id
                new_neighbours = self.region_query(pt)
                if len(new_neighbours) >= self.minPts:
                    for n in new_neighbours:
                        if self.label[n] != cluster_id:
                            neighbours.append(n)


    def region_query(self, point):
        """
        Return list of neighbours
        """
        # potential_neighbours = list(self.rtree.intersection((self.data[point][0]-self.eps, self.data[point][1]-self.eps, self.data[point][0]+self.eps, self.data[point][1]+self.eps)))
        # neighbours = []
        
        # for i in potential_neighbours:
        #     if distance.euclidean(self.data[i], self.data[point]) <= self.eps:
        #         neighbours.append(i)
        
        neighbours = self.kdtree.query_ball_point([self.data[point][0], self.data[point][1]], self.eps)
        return neighbours
    
    def preprocess_data(self):
        # d = {}
        # indexes = []
        # for id, row in self.data.iterrows():
        #     indexes.append(id)
        #     d[id] = []
        #     for col in row:
        #         d[id].append(col)
        # self.data = pd.Series(data=d, index=indexes)
        self.data = self.data.to_numpy()