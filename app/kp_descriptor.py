#! /usr/bin/python
'''
    Object which stores keypoint of feature,
    the descriptor vector, the response
'''


class KPdescriptor:
    def __init__(self, descriptors, keypoint, response):
        self.desc = descriptors
        self.kp = keypoint
        self.re = response

    def __str__(self):
        return "kp: %.2f, %.2f  |  re: %.2f" % (self.kp[0], self.kp[1], self.re)

    def log_all(self):
        # using the fact that python concatenates consecutive strings
        # don't be confused by the \
        return "keypoint: ", self.kp, \
            "\nresponse: ", self.re, \
            "\ndescriptors length: ", len(self.desc)
