import argparse
import csv
import numpy as np
import re
import heapq
import uuid

"""
AggloClustering.py by Brian Charous and Yawen Chen
An implementation of Agglomerative Clustering for the Carleton writing portfolio data

We use a set to store all the distance in heap that we should ignore when they pop. 
In this way, we decrease the amount of work finding and removing distance related to the most recently deleted centers. 

Required Parameters: k (number of clusters) f (filename containing the data)
For example: 
python AggloClustering.py -f portfoliodata.txt -k 5
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

    def sse(self):
        sse = 0
        for point in self.points:
            sse += distance(self.centroid, point)
        return sse

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
        means.append(mean)
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
            pairings.append((dist, (c1, c2)))
    return pairings

def cluster(data, k):
    """ do agglomerative clustering """
    # start off with each point being its own cluster
    clusters = {}
    for row in data:
        c = Cluster([row])
        clusters[c.id] = c
    cluster_pairings = calculate_cluster_distances(clusters.values())
    count = 0
    count2 = 0
    for t in range(len(cluster_pairings)):
        if cluster_pairings[t][0]>50:
            count+=1
            print "distance is:{0} for the {1}th comparison \n".format(cluster_pairings[t][0], t)
    print "total is:{0}\n".format(count)
    heapq.heapify(cluster_pairings)
    clusters_to_ignore = set()
    
    while len(clusters) > k:
        # ignore clusters that have already been merged
        while True:
            #closest_clusters = heapq.heappop(cluster_pairings)[1]
            closest = heapq.heappop(cluster_pairings)
            closest_clusters = closest[1]
            closest_dist = closest[0]
            c1 = closest_clusters[0]
            c2 = closest_clusters[1]
            #print "closest distance is now:{0}".format(closest_dist)
            if c1.id not in clusters_to_ignore and c2.id not in clusters_to_ignore:
                #print "\n ha, ones we should delete!"
                break
        print "***old distance is:{0} with id{1} and {2}: \n".format(closest_dist, c1.id, c2.id )

        new_closest2 = heapq.heappop(cluster_pairings)
        print "Before adding the new cluster the shortest distance is:{0} with id {1} and id {2}\n".format(new_closest2[0], new_closest2[1][0].id, new_closest2[1][1].id)
        heapq.heappush(cluster_pairings, new_closest2)
        # merge clusters
        merged_points = c1.points
        merged_points.extend(c2.points)
        merged = Cluster(merged_points)

        # remove old clusters, remember to ignore them
        del clusters[c1.id]
        del clusters[c2.id]
        clusters_to_ignore.add(c1.id)
        clusters_to_ignore.add(c2.id)

        # calculate distance between merged cluster and all others
        for cluster in clusters.values():
            dist = distance(merged.centroid, cluster.centroid)
            heapq.heappush(cluster_pairings, (dist, (merged, cluster)))
        new_closest = heapq.heappop(cluster_pairings)
        print "new distance is:{0} with id {1} and id {2}\n".format(new_closest[0], new_closest[1][0].id, new_closest[1][1].id)
        heapq.heappush(cluster_pairings, new_closest)
        if new_closest2[0] == new_closest[0]:
            count2 += 1
            print "Still old shortest path!"
        clusters[merged.id] = merged
    print "count for all new shortest is {0} \n".format(count2)
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
    total_sse = 0
    for i in range(len(centers)):
        c = centers[i]
        unstandardized_pt = []
        for j in range(len(c.centroid)):
            val = c.centroid[j]
            unstandardized = val*stdevs[j]+means[j]
            unstandardized_pt.append(unstandardized)
        total_sse += c.sse()
        print "<Center>: {0}, SSE: {1}, {2} points \n".format(unstandardized_pt, c.sse(), len(c.points))
    print "Total sse: {} \n".format(total_sse)

if __name__ == '__main__':
    main()