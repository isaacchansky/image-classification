#!/usr/bin/python
'''
Unit tests for different clustering algorithms.
'''

import sys
# append project root path (1 dir up) to the search path for modules
sys.path.append('..')
import app.fileops as fo
import app.imgproc as ip
import app.cluster as cl
import numpy as np


def initialize_testing():
    np.set_printoptions(precision=3)    # pretty printing of numpy arrays
    img_path = fo.PROJECT_ROOT_PATH + "res/img/starbucks"
    img_list = fo.read_imgs(img_path)
    gray_img_list = ip.convert_to_gray(img_list)
    print "number of images: ", len(img_list)
    computed_SURF, descriptors = ip.calculate_surf(gray_img_list, False)
    print "cs", len(computed_SURF)
    print "des ", len(descriptors)
    return computed_SURF, descriptors


def adaptive_clustering_test(computed_SURF):
    print "starting clustering..."
    # pass in KPDescriptor objects
    cluster_list, centroids = cl.adaptive_cluster(computed_SURF, 0.2, 'm-estimator')
    print "created %s clusters" % len(cluster_list)
    print "with %s centroids" % len(centroids)
    cluster_list_loc = fo.store_data('clusterListTest', cluster_list)
    print 'saved cluster list loc'
    reloaded_data = fo.load_data(cluster_list_loc)
    print reloaded_data
    return reloaded_data


def efficient_adaptive_clustering_test(computed_SURF):
    print "starting clustering..."
    # pass in KPDescriptor objects
    dist_func = cl.m_estimator
    cluster_list, centroids = cl.ea_cluter(computed_SURF, 0.2, dist_func)
    print "created %s clusters" % len(cluster_list)
    print "with %s centroids" % len(centroids)
    cluster_list_loc = fo.store_data('clusterListTest', cluster_list)
    print 'saved cluster list loc'
    reloaded_data = fo.load_data(cluster_list_loc)
    print reloaded_data
    return reloaded_data


def kmeans_clustering_test(descriptors, k):
    # pass in raw descriptors
    clusters, centroids = cl.kmeans(descriptors, k)
    print "created %s clusters" % len(clusters)
    inner_variance = []
    outer_variance = np.average(np.var(centroids))
    for i, cluster in enumerate(clusters):
        #print "\n=> CLUSTER ", i
        #print "median is %s" % np.average(np.median(cluster, 0))
        #print "st dev is %s" % np.average(np.std(cluster, 0))
        #print "range of values is %s" % np.average(np.ptp(cluster, 0))
        #print "variance is %s" % np.average(np.var(cluster, 0))
        inner_variance.append(np.average(np.var(cluster, 0)))

    return np.average(inner_variance), outer_variance


def iterative_kmeans_test(descriptors):
    idx_list = cl.iterative_k_means(descriptors)
    print idx_list

if __name__ == "__main__":
    computed_SURF, descriptors = initialize_testing()
    #adaptive_clustering_test(computed_SURF)
    '''
    maxclusters = len(descriptors)
    metric = []
    for i in range(1, maxclusters+6):
        iv, ov = kmeans_clustering_test(descriptors, i)
        metric.append(ov/iv)
    print metric
    print max(metric)
    #'''

    '''
    ikm_result = iterative_kmeans_test(descriptors)
    fo.store_data('iterative_kmeans_result', ikm_result)
    print "stored."
    r = fo.load_data('iterative_kmeans_result')
    print r
    #'''

    res = efficient_adaptive_clustering_test(computed_SURF)
