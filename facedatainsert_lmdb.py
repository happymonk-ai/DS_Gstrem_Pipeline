import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
from pathlib import Path
import glob
import os
import subprocess as sp

env = lmdb.open('DS_Gstrem_Pipeline/lmdb/face-detection.lmdb',
                max_dbs=10, map_size=int(100e9))

known_db = env.open_db(b'white_list')
unknown_db = env.open_db(b'black_list')

def conv_img2bytes(image_path):
    image  = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    encodedNumpyData = cv2.imencode('.jpg', image)[1].tobytes()
    return encodedNumpyData

def insert_db(memberId, person_img_bytes, db_):
    with env.begin(write=True) as txn:
        txn.put(bytearray(memberId, "utf-8"), person_img_bytes, db=db_)
    return True

def cid_to_image(cid):
    print(cid)
    command = 'ipfs --api=/ip4/216.48.181.154/tcp/5001 get {cid}'.format(cid=cid)
    print(command)
    output = sp.getoutput(command)
    print(output)
    image_path = "DS_Gstrem_Pipeline/image/"+str(cid)+".jpg"
    os.rename(cid, image_path)
    return image_path

def add_member_to_lmdb(MemberPublish):
    list_of_members =  MemberPublish["member"]
    for each_member in list_of_members:
        faceCID = each_member["faceCID"]
        memberId = each_member["memberId"]
        class_type = each_member["type"]
        image_path = cid_to_image(faceCID[0])
        person_img_bytes = conv_img2bytes(image_path)
        if class_type == "known":
            db_ = known_db
        else:
            db_ = unknown_db
        status  = insert_db(memberId, person_img_bytes, db_)
        print(status)
        return status

# # # #get json from nats and call add_member_to_lmdb(nats_json)
# MemberPublish = {'id': 'ui75LlKf6gzrfa7LuU2y27Jaq1nxO2nc', 'member': [{'memberId': 'did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==', 'type': 'FACEID', 'faceCID': ['Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP'], 'role': 'admin'}]}
# print(add_member_to_lmdb(MemberPublish))


