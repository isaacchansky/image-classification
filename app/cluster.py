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


def mahalanobis(a, b, threshold):
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


def adaptive_cluster(structured_data, dist_threshold, distance_metric):
    '''
     data is an array of img definitions
     img definitions being arrays of KPDescriptor objects
       so data is n x m(#imgs) x 64
    '''
    start_time = datetime.datetime.now()
    cluster_list = []    # variable size list of clusters
    numImgs = len(structured_data)
   # print structured_data
    for idx, img in list(enumerate(structured_data)):
        if idx > 1:
            print "tested against 1 images"
            break
        for kp_obj in img:  # for each kp_obj in initial image
           # print "for each kp_obj in img..."
           # print kp_obj
            descriptor = kp_obj.desc
            potential_cluster = []   # working array of descriptors (np.arrays)
            # compare each kp_obj to every other one in other imgs
            for c_idx, c_img in list(enumerate(structured_data)):
               # print "for each other img in data..."
               # print c_idx," = ",idx
                if c_idx != idx:    # dont want to compare within same image
                    d = imgproc.get_all_descriptors(c_img)
                    closest_index = find_closest(d, descriptor)
                    closest_desc = d[closest_index]

                    # use specified distance metric
                    if distance_metric.lower() == 'euclidian':
                        if(euclidian(descriptor, closest_desc) < dist_threshold):
                          #  print "pretty close! add it to the potential cluster"
                            potential_cluster.append(closest_desc)
                        #else:
                           # print "mehh, I've seen closer..."

                    elif distance_metric.lower() == 'mahalanobis':
                        if(mahalanobis(descriptor, closest_desc) < dist_threshold):
                          #  print "pretty close! add it to the potential cluster"
                            potential_cluster.append(closest_desc)
                        #else:
                           # print "mehh, I've seen closer..."

            #after looking at every other image and adding closest...
            if(len(potential_cluster) > numImgs/2):
                # we found a 'close' descriptor in 50%+ of images we looked at
                cluster_list.append(potential_cluster)
           # else:
               # print "threw out potential cluster..."

    end_time = datetime.datetime.now()
    print "done, took ", (end_time - start_time)
    return cluster_list


# straight average of points, probably kinda bad..
def create_centroids(cluster_list):
    #input -> list of points/vectors making up cluster
    #return centroid

    centroid_list = []

    for cluster in cluster_list:
        #cluster is a list of descriptors
        c = np.matrix(cluster)  # create matrix
        centroid = c.mean(0)    # centroid/mean vector
        centroid_list.append(centroid)
    return centroid_list


#more accurate centroid method -> convex hull peeling
# get the outliers and throw them out... do this a few times? until the cluster is halved?


'''
K-means

input - descriptors, K value
'''


def kmeans(descriptors, K):
    data = np.array(imgproc.reshape_descriptors_for_db(descriptors))
    cluster_data = vq.kmeans2(data, K)
    centroids = cluster_data[0]
    labels = cluster_data[1]

    clusterList = [[0] for i in range(K)]
    for label in labels:
        clusterList[label].append(data[label])

    return clusterList, centroids


'''
Iterative k-means.
input - descriptors, max K value
'''


def iter_kmeans(descriptors, max_K):
    data = np.array(imgproc.reshape_descriptors_for_db(descriptors))

    np.set_printoptions(precision=3)    # pretty printing of numpy arrays

    for k in range(max_K, -1):  # count down from max
        clusters, centroids = kmeans(data, max_K)
