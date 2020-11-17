import os
import pickle

CACHE_DIR = '../cache'

def listcache(q = None):
    objnames = [fname.split('.')[0] for fname in os.listdir(CACHE_DIR)]
    if q:
        objnames = [objname for objname in objnames if q in objname]
    return objnames

def writecache(objname, obj):
    with open(f'{CACHE_DIR}/{objname}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def readcache(objname):
    try:
        with open(f'{CACHE_DIR}/{objname}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        print(e)
        return None

def purgecache(objname):
    return os.remove(f'{CACHE_DIR}/{objname}.pkl')

