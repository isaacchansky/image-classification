#! /usr/bin/python

'''
Author: Isaac Chansky

Utility functions and functions for
multiple approaches to clustering.

All cluster functions should be of the following format:
    input   -> a list of imgs(= a list of kp_descriptor objects)
    output  -> a list of clusters(= list of descriptor vectors)

'''

import datetime
import numpy as np
import imgproc
import scipy.cluster.vq as vq

'''
    UTILITY FUNCTIONS
'''


def find_closest(data, vect):
    disp = data - vect
    # returns index where it finds the closest point
    return np.argmin((disp*disp).sum(1))


def euclidian(a, b):
    # a and b are n-dimensional arrays of floats
    running_sum = 0
    for i in range(len(a)):
        running_sum += (a[i] - b[i]) ** 2
    result = running_sum ** .5
    return result


def m_estimator(a, b, threshold):
    # a and b are n-dimensinoal arrays of floats
    running_sum = 0
    for i in range(len(a)):
        dist = a[i] - b[i]
        if abs(dist) > threshold:
            dist = threshold
        running_sum += dist**2
    result = running_sum ** .5
    return result


'''
    CLUSTER FUNCTIONS
'''


def adaptive_cluster(structured_data, dist_threshold, dist_metric):
    '''
     data is an array of img definitions
     img definitions being arrays of KPDescriptor objects
       so data is n x m(#imgs) x 64
    '''
    start_time = datetime.datetime.now()
    cluster_list = []    # variable size list of clusters
    numImgs = len(structured_data)
    dist_metric_thresh = 10

   # print structured_data
    for idx, img in list(enumerate(structured_data)):
        if idx > 2:
            break
        for kp_obj in img:  # for each kp_obj in initial image
            descriptor = kp_obj.desc
            potential_cluster = []   # working array of descriptors (np.arrays)
            # compare each kp_obj to every other one in other imgs
            for c_idx, c_img in list(enumerate(structured_data)):
                if c_idx != idx:    # dont want to compare within same image
                    d = imgproc.get_all_descriptors(c_img)
                    closest_index = find_closest(d, descriptor)
                    closest_desc = d[closest_index]

                    # use specified distance metric
                    if dist_metric.lower() == 'euclidian':
                        if(euclidian(descriptor, closest_desc) < dist_threshold):
                            potential_cluster.append(closest_desc)
                    elif dist_metric.lower() == 'm-estimator':
                        if(m_estimator(descriptor, closest_desc, dist_metric_thresh) < dist_threshold):
                            potential_cluster.append(closest_desc)

            #after looking at every other image and adding closest...
            if(len(potential_cluster) > numImgs/2):
                # we found a 'close' descriptor in 50%+ of images we looked at
                cluster_list.append(potential_cluster)

        iter_end_time = datetime.datetime.now()
        print "finished image, took ", (iter_end_time - start_time)

    print "generating centroids..."
    centroids = create_centroids(cluster_list)
    end_time = datetime.datetime.now()
    print "done, took ", (end_time - start_time)
    #recluster_centroids(centroids)
    return cluster_list, centroids


# straight average of points, probably kinda bad..
def create_centroids(cluster_list):
    #input -> list of points/vectors making up cluster
    #return centroid
    centroid_list = []
    for cluster in cluster_list:
        #cluster is a list of descriptors
        c = np.array(cluster)
        centroid = np.mean(c, 0)    # centroid/mean vector
        centroid_list.append(centroid)
    return centroid_list


def recluster_centroids(centroid_list):
    # function used to consolidate the many centroids after initial adaptive clustering
    clusters = []

    for centroid in centroid_list:
        print "do something..."
    print len(centroid_list)
    print centroid_list[0]

    # ^^ maybe just use iterative k-means for this business? guarenteed a smaller data set than before...


def iterative_k_means(unstructured_data):
    k_clusters_list = []    # list of eventual length k, holding respective clusters formed as tuples of (cluster_list, centroids)
    k_step = 2
    data = np.array(imgproc.reshape_descriptors_for_db(unstructured_data))
    for K in range(len(unstructured_data)):
        cluster_data = vq.kmeans2(data, K)
        K += k_step
        centroids = cluster_data[0]
        labels = cluster_data[1]
        clusterList = []
        for label in labels:
            clusterList.append(data[label])
        k_clusters_list.append(clusterList, centroids)
    # now I have a k-length list of each group of clusters...

    # so compute the Calinski-Harabasz index ( inter-cluster dist / within-cluster-dist) for each k
    inter_cluster_dists = []    # list of sum of distances between centroids for each k
    inner_cluster_dists = []    # list of avg inner cluster distances for each k
    #for item in k_clusters_list:    # item is (cluster_list, centroids)

    # FINISH UP!


def kmeans(descriptors, K):
    data = np.array(imgproc.reshape_descriptors_for_db(descriptors))
    cluster_data = vq.kmeans2(data, K)
    centroids = cluster_data[0]
    labels = cluster_data[1]

    clusterList = [[] for i in range(K)]
    for label in labels:
        clusterList[label].append(data[label])

    return clusterList, centroids
