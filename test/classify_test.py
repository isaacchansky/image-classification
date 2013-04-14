#!/usr/bin/python
'''
Testing the full classification system...
'''

import sys
# append project root path (1 dir up) to the search path for modules
sys.path.append('..')
import app.fileops as fo
import app.imgproc as ip
import app.cluster as cl
import app.classify as classify
import cv2
import os


root_img_path = fo.PROJECT_ROOT_PATH
img_path_dict = {
            'starbucks' : root_img_path + 'res/img/starbucks',
            'adidas'    : root_img_path + 'res/img/adidas',
            'android'   : root_img_path + 'res/img/android',
            'nike'      : root_img_path + 'res/img/nike',
            'reddit'    : root_img_path + 'res/img/reddit',
            'bbc'       : root_img_path + 'res/img/bbc'
            }

img_path = img_path_dict['starbucks']
img_list = fo.read_imgs(img_path)
gray_img_list = ip.convert_to_gray(img_list)

computed_SURF, descriptors = ip.calculate_surf(gray_img_list, False)

print "starting clustering..."
clusters, centroids = cl.adaptive_cluster(computed_SURF, 0.2, 'euclidian')

fo.store_data('clusters', clusters)
fo.store_data('centroids', centroids)

loaded_clusters = fo.load_data('clusters')
loaded_centroids = fo.load_data('centroids')
#clusters, centroids = cl.kmeans(descriptors, 25)
print "created %s clusters" % len(loaded_clusters)
print "created %s centroids of length %s" % (len(loaded_centroids), len(loaded_centroids[0]))

# get test image set up
test_img_path = img_path + "/test"
test_imgs = fo.read_imgs(test_img_path)
labels = [f for f in os.listdir(test_img_path)if f.endswith('.jpg')]

#

results = []

for i, test_img in list(enumerate(test_imgs)):
    # run SURF
    test_computed_SURF, test_descriptors = ip.single_img_surf(test_img)

    #
    # classify against clusters
    kdclassify = classify.classify_kdtree(loaded_centroids, test_descriptors)

    saveimg = test_img

    color = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (0, 255, 255),
        5: (255, 0, 255),
        6: (150, 0, 0),
        7: (0, 150, 0),
        8: (0, 0, 150),
        9: (0, 0, 0),
        10: (75, 0, 0),
        11: (0, 75, 0),
        12: (0, 0, 75),
        13: (75, 75, 0),
        14: (0, 75, 75),
        15: (75, 0, 75),
        16: (10, 0, 0),
        17: (0, 10, 0),
        18: (0, 0, 10),
        19: (10, 10, 10)
    }
    cluster_num = 0
    for j, dist in list(enumerate(kdclassify[0])):
        if dist < 0.11:
            cluster_num += 1
            rad = 10  # rad = int(test_computed_SURF[i].re / 20)
            x = int(test_computed_SURF[j].kp[0])
            y = int(test_computed_SURF[j].kp[1])
            cluster = int(kdclassify[1][j])
            cluster_label = int(kdclassify[1][j] % 20)
            #print "cluster is: " + str(cluster)
            print x, ",", y
            cv2.circle(saveimg, (x, y), rad, (0, 0, 255))  # color[cluster_label])
    results.append((labels[i], cluster_num))
    print labels[i], ":", cluster_num
    cv2.imshow('adaptive_cluster', saveimg)
    cv2.waitKey(0)
    cv2.destroyWindow('adaptive_cluster')

print results
