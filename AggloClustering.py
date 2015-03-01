import codecs
import string
import math
import random
import argparse
import sys

"""
AggloClustering.py by Brian Charous and Yawen Chen
An implementation of Agglomerative Clustering

To compile: clustering.py -k (number of clusters) -f (filename) -i (initialization method: either random or distance, default is random) 
For example: 
                     python AggloClustering.py.py -f portfoliodata.txt  -k 5
"""

def read_data(filename):
    """ add data_id, convert '   ' to null, also replace all 9999.99 to null
     return list of points from file with structure like
    (data_id, [n1, n2,n3, null, n4, n5...,n15...]), where n  are some float"""
    data = []
    data_id = 0
    with codecs.open(filename) as f:
        next(f) # skip first line
        for line in f:
            coords = []
            line_strip = string.replace(line, '\r\n', '')
            components = line_strip.split('\t')
            for i in components:    
                if i == "9999.99":
                    coords.append('null')    
                else: 
                    try:
                        coords.append(float(i))              
                    except ValueError:
                       coords.append('null')                                 
            data.append((data_id, coords))
            data_id+= 1
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True, help='Data file name')
    parser.add_argument('-k', '--clusters', type= int, required=True, help='Number of clusters')
    args = parser.parse_args()

    k = args.clusters
    filename = args.filename

    data = read_data(filename)


if __name__ == '__main__':
    main()

