#!/usr/bin/python

'''
Unit test for SURF extraction.
'''
import sys
# append project root path (1 dir up) to the search path for modules
sys.path.append('..')
import app.fileops as fo
import app.imgproc as ip

img_db_path = fo.PROJECT_ROOT_PATH + "res/img/nike"

img_list = fo.read_imgs(img_db_path)
gray_img_list = ip.convert_to_gray(img_list)

computed_SURF, descriptors = ip.calculate_surf(gray_img_list, False)

desc_list = ip.reshape_descriptors_for_db(descriptors)
print desc_list

data = ip.get_all_descriptors(desc_list)

#cluster data

# figure it out!


test_img_path = img_db_path + "/test"
test_imgs = fo.read_imgs(test_img_path)
test_img = test_imgs[1]

test_img_kpd, all_test_desc = ip.calculate_surf(test_img)

