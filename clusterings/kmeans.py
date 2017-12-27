"""
Created by Ramdani (ramd4ni@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt

from random import choice, sample, random
from scipy.spatial import distance


class Kmeans:

    def __init__(self, k_cluster, points, initial_k=None):
        """
        :param k_cluster: set number of cluster
        :param initial_k: initial centroid
        """
        self.clusters = {}
        self.k = k_cluster
        self.points = points
        self.initial_k = initial_k
        self.centroids = []
        self.maximum_loop = 100


    def calculate_sum_square(self):
        """
            calculate Sum of Squared Errors
        :return:
        """
        metrics = []
        for icluster in self.clusters.keys():
            m = 0
            for icen in range(len(self.centroids)):
                for point in self.clusters[icluster]:
                    t = distance.euclidean(point, self.centroids[icen])
                    m = m + round(t, 2)
            metrics.append(1 / m)
        return metrics

    def run(self, plot=False, is_animation=False, pause_time=5, verbose=False, centroids=None):
        """
            X is list in list
            [[1,2], [3,4], [5,6], [7,8]]
        :param x: `list`
        :return:
        """
        i = 0
        stop = False
        self.centroids = self.select_initial_k(centroids)

        if plot:
            self.plot('initial.png')

        sse = []

        while i < self.maximum_loop and not stop:

            self.compute_each_point_to_centrois()

            self.plot('fig{}.png'.format(i))

            new_centroids = self.recalculate_centroids_based_on_cluster()

            sse.append(self.calculate_sum_square())

            if new_centroids != self.centroids:
                self.centroids = new_centroids
                stop = False
            else:
                stop = True

            i += 1

            if verbose:
                print("centroids: {}".format(self.centroids))

        if plot:
            self.plot("result.png".format(i))
            if is_animation:
                plt.pause(pause_time)

        if verbose:
            print("Total Iteration: {}".format(i))
            # print("SSE: {}".format(sse))
            print("Total SSE: {}".format(len(sse)))
            for s in sse:
                print(s)

        if plot:
            plt.show()

    def select_initial_k(self, centroids=None):
        """
            select initial centroid randomly. can override using the other method
        :return:
        """
        if centroids is None:
            centroids = []
            loop = self.k
            while loop > 0:
                c = choice(self.points)
                if c not in centroids:
                    centroids.append(c)
                else:
                    loop += 1
                loop -= 1

        return centroids

    def calculate_distance_point_to_each_centroid(self, point):
        """
        :param point:
        :return: returning list of distance
        """
        return [distance.euclidean(point, centroid) for centroid in self.centroids]

    def recalculate_centroids_based_on_cluster(self):
        """
            recalculate centroids according cluster
        :return: new centroid
        """
        new_centroids = []
        for points in self.clusters.values():
            total_x = 0
            total_y = 0
            for point in points:
                total_x += point[0]
                total_y += point[1]

            new_centroids.append(list([total_x/len(points), total_y/len(points)]))

        return new_centroids

    def compute_each_point_to_centrois(self):
        """
            compute each point to centroids and assign according nearest centroid
        :return:
        """
        clusters = {}
        for point in self.points:
            distances = self.calculate_distance_point_to_each_centroid(point)

            shortest_cluster_index = np.argmin(distances)

            if shortest_cluster_index in clusters:
                clusters[shortest_cluster_index].append(point)
            else:
                clusters[shortest_cluster_index] = [point]

        self.clusters = clusters

    @property
    def labels(self):
        return [cluster for cluster in self.clusters for points in self.clusters[cluster]]

    @property
    def X(self):
        return [points[0] for cluster in self.clusters.values() for points in cluster]

    @property
    def Y(self):
        return [points[1] for cluster in self.clusters.values() for points in cluster]

    def plot(self, name_file=None):
        title = "K={} \n N={}".format(len(self.centroids), len(self.points))
        plt.title(title, fontsize=16)

        # x, y, labels = add_center_point(self.X, self.Y, self.labels, self.centroids)
        c_x = []
        c_y = []

        for point in self.centroids:
            c_x.append(point[0])
            c_y.append(point[1])

        plt.cla()
        plt.scatter(self.X, self.Y, c=self.labels, alpha=0.8, s=1)
        plt.scatter(c_x, c_y,  marker='*', c='#FF0000')

        plt.savefig(name_file)

        plt.draw()


def add_center_point(X, Y, lables, centroids):
    X_, Y_, labels_ = X, Y, lables
    for centroid in centroids:
        X_.append(centroid[0])
        Y_.append(centroid[1])
        labels_.append(20)

    return X_, Y_, labels_


class KmeansPlusPlus(Kmeans):

    def _dist_from_centers(self):
        cent = self.centroids
        X = self.points
        D2 = np.array([min([distance.euclidean(x,c) ** 2 for c in cent]) for x in X])
        self.D2 = D2

    def _choose_next_center(self):
        self.probs = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return (self.points[ind])

    def select_initial_k(self, centroids=None):
        self.centroids = sample(self.points, 1)
        i=0
        while len(self.centroids) < self.k:
            self._dist_from_centers()
            self.centroids.append(self._choose_next_center())
            i += 1
        self.centroids.sort(key=lambda x: x[0])
        return self.centroids



if __name__ == '__main__':
    data = input()
    total, dimension, k_cluster = data.split(" ")
    raw_data = []
    for i in range(int(total)):
        a = input()
        raw_data.append(list([float(j) for j in a.split()]))

    kmeans = Kmeans(int(k_cluster), raw_data)
    kmeans.run(plot=True, is_animation=False, verbose=True, pause_time=1)
