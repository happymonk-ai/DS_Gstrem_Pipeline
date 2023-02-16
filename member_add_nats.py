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
import nats

from nats.aio.client import Client as NATS

from facedatainsert_lmdb import add_member_to_lmdb

nc = NATS()



async def cb(msg):
    try :
        data = (msg.data)
        print(data)
        data  = data.decode()
        print(data)
        data = json.loads(data)
        print(data)
        status = add_member_to_lmdb(data)
        if status:
            subject = msg.subject
            reply = msg.reply
            data = msg.data.decode()
            await nc.publish(msg.reply,b'ok')
            print("Received a message on '{subject} {reply}': {data}".format(
                subject=subject, reply=reply, data=data))

        
    except TypeError as e:
        print(TypeError," nats add member error >> ", e)
        
    finally:
        print("done with work ")
        # sem.release()

async def main():
    #await member_video_ipfs(member_did, member_name, member_cid)
    await nc.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    sub = await nc.subscribe("member.update.*", cb=cb)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.run_forever()


    
    # except RuntimeError as e:
    #     print("error ", e)
    #     print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")

  #  b'{"memberId":"did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==","faceid":["Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP"],"type":"ADD_FACE","createdAt":"2023-02-02T16:42:49.233+05:30"}'
#{'id': 'tuxSv3G6ljsjjM2gTr-qN0sOL7HkB37j', 'type': 'FACEID', 'member': [{'memberId': 'did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==', 'faceCID': ['Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP'], 'role': 'admin'}]}