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
from time import sleep
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


def m_estimator(a, b):
    # a and b are n-dimensinoal arrays of floats
    running_sum = 0
    threshold = 10
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
     structured_data is an array of img definitions
     img definitions being arrays of KPDescriptor objects
       so data is n(#imgs) x m(#features) x 64 (single feature)
    '''
    start_time = datetime.datetime.now()
    cluster_list = []    # list of clusters, will return
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


'''
A more efficient, less noisy adaptive clustering algorithm
pass in dist_func as literal function name
'''


def ea_cluter(structured_data, d_thresh, d_func):
    '''
     structured_data is an array of img definitions
     img definitions being arrays of KPDescriptor objects
       so data is n(#imgs) x m(#features) x 64 (single feature)
    '''
    start_time = datetime.datetime.now()
    cluster_list = []    # list of clusters, will return
    numImgs = len(structured_data)

   # print structured_data
    for idx, img in list(enumerate(structured_data)):
        print 'image ', idx
        if idx > 2:
            break
        for kp_obj in img:  # for each kp_obj in initial image

            # compare each kp_obj to every other one in other imgs
            potential_cluster = build_potential_cluster(idx, kp_obj, structured_data, d_func, d_thresh)

            pc_centroid = lambda x: np.mean(np.array(potential_cluster), 0)
            clust_centroids = create_centroids(cluster_list)
            #after looking at every other image and building potential cluster...

            # look at each existing clusters(centroids)
            for cent_i, cent in list(enumerate(clust_centroids)):

                # if this new cluster should really be part of an existing one
                if d_func(pc_centroid, cent) < d_thresh:
                    #concatenate potential new cluster with existing cluster in list
                    np.concatenate((cluster_list[cent_i], potential_cluster), 0)
                    print 'concatenating cluster {0}...'.format(cent_i)
                    break   # don't add to more than one cluster

                # if we make it here, then we might have a big cluster that is 'new'
                if(len(potential_cluster) > numImgs/2):
                    # we found a new 'close' descriptor in 50%+ of images we looked at
                    cluster_list.append(potential_cluster)

        iter_end_time = datetime.datetime.now()
        print "finished image, took ", (iter_end_time - start_time)
        start_time = iter_end_time

    print "generating centroids..."
    centroids = create_centroids(cluster_list)
    end_time = datetime.datetime.now()
    print "done, took ", (end_time - start_time)
    #recluster_centroids(centroids)
    return cluster_list, centroids



'''
potential_cluster = []   # working array of descriptors (np.arrays)
for c_idx, c_img in list(enumerate(structured_data)):
                print c_idx

                if c_idx != idx:    # dont want to compare within same image
                    d = imgproc.get_all_descriptors(c_img)
                    closest_index = find_closest(d, descriptor)
                    closest_desc = d[closest_index]
                    # use dist function
                    if dist_func(descriptor, closest_desc) < dist_threshold:
                        potential_cluster.append(closest_desc)
                else:
                    print 'breaking'

                    break
'''


def build_potential_cluster(curr_idx, kp_obj, structured_data, dist_func, d_thresh):
    potential_cluster = []
    descriptors = kp_obj.desc
    for i, img in list(enumerate(structured_data)):
        if i != curr_idx:
            d = imgproc.get_all_descriptors(img)
            closest_idx = find_closest(d, descriptors)
            closest_desc = d[closest_idx]
            if dist_func(descriptors, closest_desc) < d_thresh:
                potential_cluster.append(closest_desc)
            else:
                break
    return potential_cluster






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

    data = np.array(imgproc.flatten_descriptors(unstructured_data))
    Calinski_Harabasz_idx = []  # contains tuple (K, CH_ind_val)
    print "going to make {0} clusters".format(len(data))
    for K in range(1000, len(data)-1000, 3000):
        clusters, centroids = kmeans(data, K)
       # K += k_step
        clusterList = []
        inter_cluster_dist = np.std(centroids)
        inner_sum = 0
        for cluster in clusters:
            inner_sum += np.std(cluster)
        inner_cluster_dist = inner_sum/len(clusters)
        CH_ind = inter_cluster_dist/inner_cluster_dist
        Calinski_Harabasz_idx.append((K, CH_ind))
        print "clusters: {0} \n CH_index: {1}".format(K, CH_ind)
        k_clusters_list.append((clusterList, centroids))
    return Calinski_Harabasz_idx


def kmeans(descriptors, K):
    data = np.array(imgproc.flatten_descriptors(descriptors))
    cluster_data = vq.kmeans2(data, K)
    centroids = cluster_data[0]
    labels = cluster_data[1]

    clusterList = [[] for i in range(K)]
    for label in labels:
        clusterList[label].append(data[label])

    return clusterList, centroids
