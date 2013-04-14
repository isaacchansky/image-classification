#! /usr/bin/python

'''
Author: Isaac Chansky

Defines functions for writing/reading and moving images around.
'''

import cv2
import numpy as np
import os
import datetime
import pickle

# get path to project root, 1 dir above...
PROJECT_ROOT_PATH = '/'.join(os.getcwd().split('/')[:-1])+'/'


def read_imgs(path):
    print path
    #returns list of all imgs in path
    start_time = datetime.datetime.now()
    print "reading images in %s directory..." % path.split('/')[-1]
    path_list = [os.path.join(path, f) for f in os.listdir(path)
                 if f.endswith('.jpg')]
    # print path_list
    img_list = []
    for path in path_list:
        img_list.append(np.asarray(cv2.imread(path)[:, :]))

    end_time = datetime.datetime.now()
    print "done, took ", (end_time - start_time)
    return img_list


def store_data(name, data):
    # loc = PROJECT_ROOT_PATH+'data/'+name+'.pkl'
    loc = PROJECT_ROOT_PATH+'test/data/'+name+'.pkl'
    print "storing to "+loc
    f = open(loc, 'wb')
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return loc


def load_data(name):
    loc = PROJECT_ROOT_PATH+'test/data/'+name+'.pkl'
    print "loading from "+loc
    f = open(loc, 'rb')
    data = pickle.load(f)
    f.close()
    return data


#command line progress bar. make it do cooler shit!
def progress(x):
    out = '%s done' % x
    bs = '\b' * 1000
    print bs,
    print out,
