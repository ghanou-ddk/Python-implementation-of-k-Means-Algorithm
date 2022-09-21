import math

import numpy as np
import pandas as pd


df = pd.read_csv("iris.csv")


f1 = df['sepal.length'].values
f2 = df['sepal.width'].values
f3 = df['petal.length'].values
f4 = df['petal.width'].values
species = df['variety'].values

original_data = np.array(list(zip( species, f1, f2, f3, f4)))
data = np.array(list(zip(f1, f2, f3, f4)))

def init_centroids(k, data):

    c = []
    s = np.random.randint(low=1, high=len(data), size=k)
    while (len(s) != len(set(s))):
        s = np.random.randint(low=1, high=len(data), size=k)
    for i in s:
        c.append(data[i])
    return c

def euc_dist(a, b):

    sum = 0
    for i, j in zip(a, b):
        a = (i - j) * (i - j)
        sum = sum + a
    return math.sqrt(sum)

def cal_dist(centroids, data):

    c_dist = []
    for i in centroids:
        temp = []
        for j in data:
            temp.append(euc_dist(i, j))
        c_dist.append(temp)
    return c_dist

def perf_clustering(k, dist_table):

    clusters = []
    for i in range(k):
        clusters.append([])
    for i in range(len(dist_table[0])):
        d = []
        for j in range(len(dist_table)):
            d.append(dist_table[j][i])
        clusters[d.index(min(d))].append(i)
    return clusters

def update_centroids(centroids, cluster_table, data):

    for i in range(len(centroids)):
        if (len(cluster_table[i]) > 0):
            temp = []
            for j in cluster_table[i]:
                temp.append(list(data[j]))
            sum = [0] * len(centroids[i])
            for l in temp:
                sum = [(a + b) for a, b in zip(sum, l)]
            centroids[i] = [p / len(temp) for p in sum]

    return centroids


def check_n_stop(dist_mem, cluster_mem):


    c1 = all(x == dist_mem[0] for x in dist_mem)
    c2 = all(y == cluster_mem[0] for y in cluster_mem)
    if c1 or c2:
        print("Stop")
    return c1 or c2
global cluster_table
def kMeans(k, data, max_iterations):
    global cluster_table
    dist_mem = []
    cluster_mem = []

    centroids = init_centroids(k, data)
    distance_table = cal_dist(centroids, data)
    cluster_table = perf_clustering(k, distance_table)
    newCentroids = update_centroids(centroids, cluster_table, data)
    dist_mem.append(distance_table)
    cluster_mem.append(cluster_table)

    for i in range(max_iterations):
        distance_table = cal_dist(newCentroids, data)
        cluster_table = perf_clustering(k, distance_table)
        newCentroids = update_centroids(newCentroids, cluster_table, data)


        dist_mem.append(distance_table)
        cluster_mem.append(cluster_table)
        if len(dist_mem) > 10:
            dist_mem.pop(0)
            cluster_mem.pop(0)
            if check_n_stop(dist_mem, cluster_mem):
                print("Stopped at iteration #", i)
                break

    for i in range(len(newCentroids)):
        print("Centroid #", i, ": ", newCentroids[i])
        print("Members of the cluster: ")
        for j in range(len(cluster_table[i])):
            print(original_data[cluster_table[i][j]])
    



kMeans(3, data, 100)
