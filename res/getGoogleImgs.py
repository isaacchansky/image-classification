#!/usr/bin/python

'''
This script pulls images from Google Image search
The images are to be used to train computer vision
algorithms...
'''

import json
import os
import time
import requests
import sys
from StringIO import StringIO
from PIL import Image
from requests.exceptions import ConnectionError


def query(query_term, folder_name, path):

    BASE_URL = 'https://ajax.googleapis.com/ajax/services/search/images?' + 'v=1.0&q=' + query_term + '&start=%d'

    BASE_PATH = os.path.join(path, folder_name.replace(' ', '_'))

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print "made: " + BASE_PATH

    start = 0  # start query string parameter for pagination
    while start < 40:   # query 20 pages
        r = requests.get(BASE_URL % start)
        for image_info in json.loads(r.text)['responseData']['results']:
            url = image_info['unescapedUrl']
            try:
                image_r = requests.get(url)
            except ConnectionError, e:
                print 'could not download %s' % url
                continue

            #remove file-system path characters from name
            title = query_term.replace(' ', '_') + '_' + image_info['imageId']
            file = open(os.path.join(BASE_PATH, '%s.jpg') % title, 'w')
            try:
                Image.open(StringIO(image_r.content)).save(file, 'JPEG')
            except IOError, e:
                # throw away gifs and stuff
                print 'couldnt save %s' % url
                continue
            finally:
                file.close()

        print start
        start += 4  # 4 images per page

        time.sleep(2)  # don't be mean to google's servers

params = sys.argv
query(params[1], params[2], 'img')
