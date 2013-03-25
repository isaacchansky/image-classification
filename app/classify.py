#! /usr/bin/python

'''
Author: Isaac Chansky

A few different classifier functions...
'''

from scipy import spatial


def classify_kdtree(data, test_data):
    #build tree
    kdtree = spatial.KDTree(data)
    return kdtree.query(test_data)
