#!/usr/bin/python
import sys
# append project root path (1 dir up) to the search path for modules
sys.path.append('..')
import app.fileops as fo
import app.imgproc as ip
import app.cluster as cl
import app.classify as classify
import cv2

# create a path for each folder in img folder

# for each path, build clusters & centroids, store them
# create 'global' dictionary of names -> (cluster.pkl, centroid.pkl), store it.