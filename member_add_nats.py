#multi treading 
import asyncio
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
import subprocess as sp

from track import lmdb_known
from track import lmdb_unknown

data = {
    "did":"008ed",
    "firstName": "Nivetheni",
    "lastName": "",
    "email": "",
    "phoneNumber": "",
    "birthDate": "",
    "images": ['QmdoDjHfrpvHPqwWLeaxwCgRSMRYkS6z3V1oMbUTAWCe8U', 'QmYdkYptXCjSLdVTaMNcE4ofzzF5WjbtijekKZwYZK1bwu', 'QmYX63u9kvVBFMErKV7msjtgwNP8KMEz8KAxRyXbLqTcFq'],
    "role": "admin"
}

# Open (and create if necessary) our database environment. Must specify
# max_dbs=... since we're opening subdbs.
env = lmdb.open('/home/nivetheni/Face_detection_pipeline/lmdb1/face-testing.lmdb',
                max_dbs=10, map_size=int(100e9))

# Now create subdbs for known and unknown peole.
known_db = env.open_db(b'known')
unknown_db = env.open_db(b'unknown')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Parent Directory path
directory = '/home/nivetheni/Face_detection_pipeline/folder'

async def lmdb_push():
    for filename in os.listdir(directory):
        name = filename.split('.')
        path = os.path.join(directory,name[0])
        
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
                
            # #fetch from lmdb
            # with env.begin() as txn:
            #     buf = txn.get(person_name, db=known_db)
            
            # # Deserialization
            # print("Decode JSON serialized NumPy array")
            # decodedArrays = json.loads(buf)
            
            # finalNumpyArray = np.asarray(decodedArrays["array"])
            # # print("NumPy Array")
            # # print(finalNumpyArray)
            # print(np.shape(finalNumpyArray))
            
            count += 1

    await lmdb_known()
    await lmdb_unknown()  

async def member_video_ipfs(member_did, member_name, member_cid):
    subdir = member_name
    # Path
    path = os.path.join(directory, subdir)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for item in member_cid:
        iterator = member_cid.index(item)
        command = 'ipfs --api=/ip4/216.48.181.154/tcp/5001 get {hash} --output={path}'.format(hash=item, path=path)
        output = sp.getoutput(command)
        print(output)
        src_file = os.path.join(path, item)
        dest_name = os.path.join(path, '{did}-{iter}.jpg').format(did=member_did, iter=iterator)
        print(src_file)
        print(dest_name)
        os.rename(src_file, dest_name)

    await lmdb_push()

# async def cb(msg):
#     # global count 
#     # sem = asyncio.Semaphore(1)
#     # await sem.acquire()
#     try :
#         data = (msg.data)
#         val = json.dumps(data)
#         val2 = data.decode('utf-8')
#         print(data)
#         print(val)
#         print(val2)
# #         member_did = data['did']
# #         member_name = data['name']
# #         member_cid = data['cid']
# #         await member_video_ipfs(member_did, member_name, member_cid))
        
#     except TypeError as e:
#         print(TypeError," nats add member error >> ", e)
        
#     finally:
#         print("done with work ")
#         # sem.release()

async def main():
    member_did = data['did']
    member_name = data['firstName']
    member_cid = data['images']

    await member_video_ipfs(member_did, member_name, member_cid)

    # nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    # js = nc.jetstream()
    # await js.subscribe("member", cb=cb, stream="test_member")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")