#! /usr/bin/python

'''
Author: Isaac Chansky

Image manipulation and feature extraction functions.
'''

import cv2
import numpy as np
import datetime
import kp_descriptor as kpd


# IMAGE MANUPULATION

def convert_to_gray(img_list):
    #returns list of grayscale coverted images
    start_time = datetime.datetime.now()
    print "converting to grayscale..."
    grayscale_img_list = np.array(img_list)
    for i in range(0, len(img_list)):
        grayscale_img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
    end_time = datetime.datetime.now()
    print "done, took ", (end_time - start_time)
    return grayscale_img_list


def binarize_img(img):
    #assumes have already called imread() on jpg, but still in color
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, binary_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_img


'''FEATURE DETECTION FUNCTIONS'''


def calculate_surf(img_list, ext_flag):
    # returns a list of lists of KeyPointDescriptor objects

    '''
    openCV Object architecture
            -keypoint object has:
                'angle'
                'class_id'
                'octave'
                'pt'
                'response'
                'size'
            -descriptors are of type numpy.ndarray

            descriptors are one long numpy array,
            so reshape to n by 64 (or 128 if using that) array
    '''

    extended = ext_flag  # expects a boolean. may never use...
    start_time = datetime.datetime.now()
    print "calculating SURF features..."

    '''
    SURF usage:  cv2.SURF([hessianThreshold[,nOctaves[,nOctaveLayers[,extended[,upright]]]]])
        hessianThreshold -> threshold for 'hessian keypoint detector'
        nOctaves -> number of pyramid octaves keypoint detector uses. basically multiple gaussian blur levels
        nOctaveLayers -> number of octave layers within each octave...
        extended -> boolean: true - 128 element descriptors, false - 64
        upright -> boolean: true -dont compute orientation, false - do
    '''

    computed_img_list = []
    descriptor_DB = []
    surf = cv2.SURF(500)
    #surf = cv2.SURF(85, 4, 2, True, False)     #extended to 128-element descriptors

    for img in img_list:
        keys, desc = surf.detect(img, None, False)
        img_kpd_list = []    # array to put KPdescriptor objects into

        # reshape descriptors from one long shitty array,
        # to an actual formatted coherent array of 64 element descriptors
        descriptors = np.reshape(desc, (len(desc) / 64, 64))

        descriptor_DB.append(descriptors)

        #create a new keypoint object for each found kp,desc pair
        for i, key in list(enumerate(keys)):
            kpdesc = kpd.KPdescriptor(descriptors[i], key.pt,
                                      key.response)
            img_kpd_list.append(kpdesc)

        computed_img_list.append(img_kpd_list)

    end_time = datetime.datetime.now()
    print "done, took ", (end_time - start_time)

    return computed_img_list, descriptor_DB


def single_img_surf(img):
    # returns a list of KPdescriptor objects
    print "running SURF on test image..."
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(500)
    kpd_list = []
    keys, desc = surf.detect(gimg, None, False)
    descriptors = np.reshape(desc, (len(desc) / 64, 64))
    for i, key in list(enumerate(keys)):
        kpdesc = kpd.KPdescriptor(descriptors[i], key.pt, key.response)
        kpd_list.append(kpdesc)

    print "finished SURF, returning list of KPdescriptor objects"
    return kpd_list, descriptors


# DATA ANALYSIS FUNCTIONS

# All functions deal in NumPy Arrays


def flatten_descriptors(descriptors_set):
    #reshapes from 'list of list of descriptors' to list of descriptors
    descriptors_list = []
    for imgData in descriptors_set:
        for descriptor in imgData:
            descriptors_list.append(descriptor)
    #print 'data reshaped to: ', len(descriptors_list), ' by ', len(descriptors_list[0])
    return np.array(descriptors_list)


def get_all_descriptors(kpd_list):
    #combines all kpd obj descriptors into one np array
    all_desc = []
    for kpd in kpd_list:
        all_desc.append(kpd.desc)
    #print 'data reshaped to: ', len(all_desc), ' by ', len(all_desc[0])
    return np.array(all_desc)




