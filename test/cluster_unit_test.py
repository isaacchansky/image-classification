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
    img_path = fo.PROJECT_ROOT_PATH + "res/img/starbucks"
    img_list = fo.read_imgs(img_path)
    gray_img_list = ip.convert_to_gray(img_list)
    print "number of images: ", len(img_list)
    computed_SURF, descriptors = ip.calculate_surf(gray_img_list, False)
    return computed_SURF, descriptors


def adaptive_clustering_test(computed_SURF):
    print "starting clustering..."

    # ADAPTIVE CLUSTERING
    # pass in KPDescriptor objects
    clusters = cl.adaptive_cluster(computed_SURF, 0.2)

    print "created %s clusters" % len(clusters)
    avg = 0
    for cluster in clusters:
        avg += len(cluster)
    avg /= len(clusters)
    print "avg size is %s" % avg


def kmeans_clustering_test(descriptors):
    # KMEANS
    # pass in raw descriptors
    clusters, centroids = cl.kmeans(descriptors, 20)
    print "created %s clusters" % len(clusters)

    np.set_printoptions(precision=3)    # pretty printing of numpy arrays

    for cluster, i in enumerate(clusters):
        print "\n=> CLUSTER ", i
        print "median is %s" % np.median(cluster, 0)
        print "st dev is %s" % np.std(cluster, 0)
        print "range of values is %s" % np.ptp(cluster, 0)
        print "variance is %s" % np.var(cluster, 0)


if __name__ == "__main__":
    computed_SURF, descriptors = initialize_testing()
    adaptive_clustering_test(computed_SURF)
