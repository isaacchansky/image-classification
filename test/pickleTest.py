#! usr/bin/python
import numpy as np
import pickle as p


def pSave():
    array_length = 10000
    a = np.zeros(array_length, dtype=np.int32)
    b = np.zeros(array_length, dtype=np.int32)
    c = np.zeros(array_length, dtype=np.int32)
    d = np.zeros(array_length, dtype=np.int32)
    to_store = [a, b, c, d]
    f = open('test.pickle', 'wb')
    p.dump(to_store, f, p.HIGHEST_PROTOCOL)
    f.close()


def pLoad():
    r = p.load(open('test.pickle', 'rb'))
    print r

pSave()
print "saved"
pLoad()