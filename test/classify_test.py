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

img_path = fo.PROJECT_ROOT_PATH + "res/img/starbucks"
img_list = fo.read_imgs(img_path)
gray_img_list = ip.convert_to_gray(img_list)

computed_SURF, descriptors = ip.calculate_surf(gray_img_list, False)

print "starting clustering..."
# pass in KPDescriptor objects
#clusters = cl.adaptive_cluster(computed_SURF, 0.2)

clusters, labels = cl.kmeans(descriptors, 20)
print "created %s clusters" % len(clusters)


# get test image set up
test_img_path = img_path + "/test"
test_img = fo.read_imgs(test_img_path)[0]
#

# run SURF
test_computed_SURF, test_descriptors = ip.single_img_surf(test_img)
#

# classify against clusters
kdclassify = classify.classify_kdtree(clusters, test_descriptors)

saveimg = test_img

color = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
    6: (100, 0, 0),
    7: (0, 100, 0),
    8: (0, 0, 100),
    9: (0, 0, 0)
}

for i, dist in list(enumerate(kdclassify[0])):
    if dist < 0.2:
        rad = int(test_computed_SURF[i].re / 10)
        x = int(test_computed_SURF[i].kp[0])
        y = int(test_computed_SURF[i].kp[1])
        cluster = int(kdclassify[1][i] % 10)
        print "cluster is: " + str(cluster)

        cv2.circle(saveimg, (x, y), rad, color[cluster])


cv2.imshow('test', saveimg)
cv2.waitKey(0)
