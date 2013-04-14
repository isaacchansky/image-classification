#! /usr/bin/python

import sys
# append project root path (1 dir up) to the search path for modules
sys.path.append('..')
import os
from PIL import Image


def resize(imgdir, MAX_WIDTH):
    the_path = os.getcwd()+'/img/'+imgdir
    MAX_HEIGHT = MAX_WIDTH
    path_list = [os.path.join(the_path, f) for f in os.listdir(the_path) if f.endswith('.jpg')]
    for i, imgpath in list(enumerate(path_list)):
        try:
            im = Image.open(imgpath)
            s = im.size
            ratio = 1
            #change:    get the max of height or width, use that ratio
            if s[0] > MAX_WIDTH:
                ratio = MAX_WIDTH/s[0]
            if s[1] > MAX_HEIGHT:
                ratio = MAX_HEIGHT/s[1]
            w = int(s[0]*ratio)
            h = int(s[1]*ratio)
            print 'new dim: ', h, 'x', w
            newimg = im.resize((w, h), Image.ANTIALIAS)
            newpath = '/'.join(imgpath.split('/')[:-1]) + "/resize_"+imgpath.split('/')[-2]+str(i)+".jpg"
            print newpath
            newimg.save(newpath, "JPEG")
        except IOError:
            print "cannot resize image %s" % imgpath


if __name__ == '__main__':

    user_path = sys.argv[1]
    resize(user_path, 600.0)
