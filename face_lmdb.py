import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
from pathlib import Path
import glob
import os
import PIL
import ipfsApi

# Open (and create if necessary) our database environment. Must specify
# max_dbs=... since we're opening subdbs.
env = lmdb.open('/app/lmdb/face-detection.lmdb',
                max_dbs=10, map_size=int(100e9))

# Now create subdbs for known and unknown peole.
known_db = env.open_db(b'known')
unknown_db = env.open_db(b'unknown')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


#for multiple folders with images 

#directory = "./knowface"
directory = "face"  # Loading the Folder can contain (Imgeas , videos , etc)
for filename in os.listdir(directory):
    
    name = filename.split('.')

    path = os.path.join(directory,name[0])
    # path = glob.glob(path)
    # print(path)
    count =0


    for img in os.listdir(path):
        img = os.path.join(path ,img)
        print(img)
        image  = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # cv_img.append(image)


        # Serialization
        numpyData = {"array": image}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dumps() to write array into file
        # encodedNumpyData = json.dumps(numpyData)
        print("Printing JSON serialized NumPy array")
        # print(encodedNumpyData)

        #push to lmdb
        person_name = bytearray(name[0]+ str(count), "utf-8")
        person_img = bytearray(encodedNumpyData, "utf-8")
        with env.begin(write=True) as txn:
            txn.put(person_name, person_img, db=known_db)

        count += 1


#directory = "./unknowface"
directory1 = "face1"  # Loading the Folder can contain (Imgeas , videos , etc)
for filename in os.listdir(directory1):
    
    name = filename.split('.')

    path = os.path.join(directory1,name[0])
    # path = glob.glob(path)
    # print(path)
    count =0


    for img in os.listdir(path):
        img = os.path.join(path ,img)
        print(img)
        image  = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # cv_img.append(image)


        # Serialization
        numpyData = {"array": image}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dumps() to write array into file
        # encodedNumpyData = json.dumps(numpyData)
        print("Printing JSON serialized NumPy array")
        # print(encodedNumpyData)

        #push to lmdb
        person_name = bytearray(name[0]+ str(count), "utf-8")
        person_img = bytearray(encodedNumpyData, "utf-8")
        with env.begin(write=True) as txn:
            txn.put(person_name, person_img, db=unknown_db)
        count += 1



