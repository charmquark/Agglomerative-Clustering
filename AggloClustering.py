import string
import math
import argparse
import csv
import numpy as np
import re
import heapq
import uuid

"""
AggloClustering.py by Brian Charous and Yawen Chen
An implementation of Agglomerative Clustering

To compile: clustering.py -k (number of clusters) -f (filename) -i (initialization method: either random or distance, default is random) 
For example: 
python AggloClustering.py.py -f portfoliodata.txt  -k 5
"""

class Cluster(object):

    def calculate_centroid(self):
        if len(self.points) > 0:
            avg = [0] * len(self.points[0])
            for point in self.points:
                for i, elem in enumerate(point):
                    avg[i] += elem
            avg = [i/len(self.points) for i in avg]
            self.centroid = avg

    def __repr__(self):
        return '<Cluster> center at {}'.format(self.centroid)

    def __init__(self, points = []):
        super(Cluster, self).__init__()
        self.points = points
        self.centroid = None
        self.id = uuid.uuid4()
        self.calculate_centroid()

def get_data(filename):
    """ read and standardize data from file """
    rows = []
    with open(filename, 'r') as f:
        next(f)
        reader = csv.reader(f, delimiter='\t')
        # replace empty field or 9999.99 with None
        junk = re.compile('^\s*$|^9999.99$')
        for row in reader:
            for i in range(len(row)):
                if junk.match(row[i]):
                    row[i] = None
                else:
                    row[i] = float(row[i])
            rows.append(row)

    # transpose to make it easier to compute mean, stdev
    columns = np.matrix.transpose(np.array(rows))
    standardized_columns = []
    stdevs = []
    means = []
    for column in columns:
        # ignore empty fields for now
        actual_vals = [v for v in column if v is not None]
        sigma = np.std(actual_vals)
        mean = np.mean(actual_vals)
        stdevs.append(sigma)
        means.append(sigma)
        # standardize if value not empty, else replace with 0
        standardized_columns.append([(x - mean)/sigma if x is not None else 0 for x in column])
    # turn back into original form
    standardized_rows = np.matrix.transpose(np.array(standardized_columns))
    return standardized_rows.tolist(), stdevs, means

def distance(p1, p2):
    """ euclidean^2 distance between 2 points """
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i]-p2[i])**2
    return distance

def calculate_cluster_distances(clusters):
    pairings = []
    for i in range(len(clusters)):
        c1 = clusters[i]
        for j in range(i+1, len(clusters)):
            c2 = clusters[j]
            dist = distance(c1.centroid, c2.centroid)
            pairings.append((dist, c1, c2))
    return pairings

def cluster(data, k):
    """ do agglomerative clustering """
    # start off with each point being its own cluster
    clusters = {}
    for row in data:
        c = Cluster([row])
        clusters[c.id] = c
    cluster_pairings = calculate_cluster_distances(clusters.values())
    heapq.heapify(cluster_pairings)
    clusters_to_ignore = set()
    while len(clusters) > k:
        # ignore clusters that have already been merged
        while True:
            closest_clusters = heapq.heappop(cluster_pairings)
            c1 = closest_clusters[1]
            c2 = closest_clusters[2]
            if c1.id not in clusters_to_ignore and c2.id not in clusters_to_ignore:
                break

        # merge clusters
        merged_points = c1.points
        merged_points.extend(c2.points)
        merged = Cluster(merged_points)
        # print c1.id, c2.id, merged.id

        # remove old clusters, remember to ignore them
        del clusters[c1.id]
        del clusters[c2.id]
        clusters_to_ignore.add(c1.id)
        clusters_to_ignore.add(c2.id)

        # calculate distance between merged cluster and all others
        for cluster in clusters.values():
            dist = distance(merged.centroid, cluster.centroid)
            heapq.heappush(cluster_pairings, (dist, merged, cluster))

        clusters[merged.id] = merged

    return clusters.values()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True, help='Data file name')
    parser.add_argument('-k', '--clusters', type= int, required=True, help='Number of clusters')
    args = parser.parse_args()

    filename = args.filename
    k = args.clusters

    data, stdevs, means = get_data(filename)
    centers = cluster(data, k)
    for i in range(len(centers)):
        c = centers[i]
        unstandardized_pt = []
        for j in range(len(c.centroid)):
            val = c.centroid[j]
            unstandardized = val*stdevs[j]+means[j]
            unstandardized_pt.append(unstandardized)
        print "<Center> {}".format(unstandardized_pt)

if __name__ == '__main__':
    main()