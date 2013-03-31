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
root_img_path = fo.PROJECT_ROOT_PATH
img_path_dict = {
            'starbucks' : root_img_path + 'res/img/starbucks',
            'adidas'    : root_img_path + 'res/img/adidas',
            'android'   : root_img_path + 'res/img/android',
            'nike'      : root_img_path + 'res/img/nike',
            'reddit'    : root_img_path + 'res/img/reddit'
            }

img_path = img_path_dict['starbucks']
img_list = fo.read_imgs(img_path)
gray_img_list = ip.convert_to_gray(img_list)

computed_SURF, descriptors = ip.calculate_surf(gray_img_list, False)

print "starting clustering..."
# pass in KPDescriptor objects
#clusters = cl.adaptive_cluster(computed_SURF, 0.2)

clusters, centroids = cl.adaptive_cluster(computed_SURF, 0.5, 'm-estimator')
#clusters, centroids = cl.kmeans(descriptors, 25)
print "created %s clusters" % len(clusters)
print "created %s centroids of length %s" % (len(centroids), len(centroids[0]))

# get test image set up
test_img_path = img_path + "/test"
test_img = fo.read_imgs(test_img_path)[0]
#

# run SURF
test_computed_SURF, test_descriptors = ip.single_img_surf(test_img)
#


# classify against clusters
kdclassify = classify.classify_kdtree(centroids, test_descriptors)

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

for i, dist in list(enumerate(kdclassify[0])):
    if dist < 0.2:
        rad = 10  # rad = int(test_computed_SURF[i].re / 20)
        x = int(test_computed_SURF[i].kp[0])
        y = int(test_computed_SURF[i].kp[1])
        cluster = int(kdclassify[1][i])
        cluster_label = int(kdclassify[1][i] % 20)
        print "cluster is: " + str(cluster)

        cv2.circle(saveimg, (x, y), rad, color[cluster_label])


cv2.imshow('adaptive_cluster', saveimg)
cv2.waitKey(0)
