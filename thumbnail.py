# gstreamer
import sys
from io import BytesIO
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

#multi treading 
import asyncio
import nats
import os
import json
import numpy as np 
from PIL import Image
import cv2
import glob
from nanoid import generate
from multiprocessing import Process, Queue
import torch
import torchvision.transforms as T
from general import (check_requirements_pipeline)
import logging 
import threading
import gc
import datetime #datetime module to fetch current time when frame is detected
import shutil
import ast
from nats.aio.client import Client as NATS
import nats

#Detection
from track_test import run
# from track import lmdb_known
# from track import lmdb_unknown

#PytorchVideo
from functools import partial

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slow_r50_detection # Another option is slowfast_r50_detection

from visualization import VideoVisualizer

from pytz import timezone
from datetime import datetime 
import imageio
import subprocess as sp

# face_detection
# import lmdb

path = "./Nats_output"
hls_path = "./Hls_output"
gif_path = "./Gif_output"

if os.path.exists(path) is False:
    os.mkdir(path)
    
if os.path.exists(hls_path) is False:
    os.mkdir(hls_path)
    
if os.path.exists(gif_path) is False:
    os.mkdir(gif_path)

start_flag = True

image_count = 0
frames = []
  
# Multi-threading
TOLERANCE = 0.62
MODEL = 'cnn'
count_person =0
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
face_did_encoding_store = dict()
track_type = []
dict_frame = {}
frame = []
count_frame ={}
count = 0
processes = []
devicesUnique = []
activity_list = []
detect_count = []
person_count = []
vehicle_count = []
avg_Batchcount_person = 0
avg_Batchcount_vehicel = 0
activity_list= []
geo_locations = []
track_person = []
track_vehicle = []
track_elephant = []
batch_person_id = []
detect_img_cid = ''
avg_Batchcount_elephant = 0
null = None

gst_str = ''

queue1 = Queue()
queue2 = Queue()
queue3 = Queue()
queue4 = Queue()
queue5 = Queue()
queue6 = Queue()
queue7 = Queue()
queue8 = Queue()
queue9 = Queue()
queue10 = Queue()
queue11 = Queue()

device = 'cuda' # or 'cpu'
video_model = slow_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)

# gstreamer
# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)
image_arr = None

device_types = ['', 'h.264', 'h.264', 'h.264', 'h.265', 'h.264', 'h.265', 'mp4', 'mp4', 'mp4', 'mp4', 'mp4', 'mp4', 'mp4']
load_dotenv()

nc_client = NATS()

# activity
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
count_video = 0 


async def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes

async def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )
    
    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )
    
    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )
    
    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), ori_boxes

async def Activity(source,device_id,source_1):
    global vide_save_path, avg_Batchcount_person, avg_Batchcount_vehicel,avg_Batchcount_elephant,track_person,track_vehicle,track_elephant,detect_count,detect_img_cid,track_dir,track_type,batch_person_id

    # Create an id to label name mapping
    global count_video            
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
    # Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
    video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.5)
    
    encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(source)
    
    time_stamp_range = range(1,2) # time stamps in video for which clip is sampled. 
    clip_duration = 2.0 # Duration of clip used for each inference step.
    gif_imgs = []
    
    for time_stamp in time_stamp_range:    
        # print("Generating predictions for time stamp: {} sec".format(time_stamp))
        
        # Generate clip around the designated time stamps
        inp_imgs = encoded_vid.get_clip(
            time_stamp - clip_duration/2.0, # start second
            time_stamp + clip_duration/2.0  # end second
        )
        inp_imgs = inp_imgs['video']
        
        # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
        # We use the the middle image in each clip to generate the bounding boxes.
        inp_img = inp_imgs[:,inp_imgs.shape[1]//2,:,:]
        inp_img = inp_img.permute(1,2,0)
        
        # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
        predicted_boxes = await get_person_bboxes(inp_img, predictor) 
        if len(predicted_boxes) == 0: 
            # print("Skipping clip no frames detected at time stamp: ", time_stamp)
            continue
            
        # Preprocess clip and bounding boxes for video action recognition.
        inputs, inp_boxes, _ = await ava_inference_transform(inp_imgs, predicted_boxes.numpy())
        # Prepend data sample id for each bounding box. 
        # For more details refere to the RoIAlign in Detectron2
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
        
        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(device)
        preds = video_model(inputs, inp_boxes.to(device))

        preds= preds.to('cpu')
        # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
        preds = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)
        
        # Plot predictions on the video and save for later visualization.
        inp_imgs = inp_imgs.permute(1,2,3,0)
        inp_imgs = inp_imgs/255.0
        out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
        gif_imgs += out_img_pred
    try:
        height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]
        vide_save_path = path+'/'+str(device_id)+'/'+str(count_video)+'_activity.mp4'
        video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))
    
        for image in gif_imgs:
            img = (255*image).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.write(img)
        video.release()
        
        await asyncio.sleep(1)
        det = Process(target= run(source=vide_save_path, queue1=queue1))
        det.start()
        batch_output = queue1.get()

        avg_Batchcount_person = batch_output["avg_person_count"]
        avg_Batchcount_vehicel= batch_output["avg_vehicle_count"]
        detect_count = batch_output["detect_count"] 
        track_person = batch_output["track_person"]
        track_vehicle = batch_output["track_vehicle"]
        detect_img_cid = batch_output["detect_img_cid"] 
        track_dir = batch_output["save_dir"] 
        track_type = batch_output["track_type"] 
        batch_person_id = batch_output["batch_person_id"] 
        track_elephant = batch_output["track_elephant"] 
        avg_Batchcount_elephant = batch_output["avg_elephant_count"]
        
    except IndexError:
        print("No Activity")
        # activity_list.append("No Activity")
        open('classes.txt','w')
        await asyncio.sleep(1)
        det = Process(target= run(source=source_1, queue1=queue1))
        det.start()
        batch_output = queue1.get()

        avg_Batchcount_person = batch_output["avg_person_count"]
        avg_Batchcount_vehicel= batch_output["avg_vehicle_count"]
        detect_count = batch_output["detect_count"] 
        track_person = batch_output["track_person"]
        track_vehicle = batch_output["track_vehicle"]
        detect_img_cid = batch_output["detect_img_cid"] 
        track_dir = batch_output["save_dir"] 
        track_type = batch_output["track_type"] 
        batch_person_id = batch_output["batch_person_id"] 
        track_elephant = batch_output["track_elephant"] 
        avg_Batchcount_elephant = batch_output["avg_elephant_count"]
    count_video += 1


async def BatchJson(source):
    global activity_list ,activity_list_box , person_count
    # We open the text file once it is created after calling the class in test2.py
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
    # We open the text file once it is created after calling the class in test2.py
    file =  open(source, 'r')
    if file.mode=='r':
        contents= file.read()
    # Read activity labels from text file and store them in a list
    label = []
    # print('Content length: ', len(contents))
    for ind,item in enumerate(contents):
        if contents[ind]=='[' and contents[ind+1] == '[':
            continue
        if contents[ind]==']':
            if ind == len(contents)-1:
                break
            else:
                ind += 3
                continue
        if contents[ind]=='[' and contents[ind+1] != '[':
            ind += 1
            if ind>len(contents)-1:
                break
            label_each = []
            string = ''
            while contents[ind] != ']':
                if contents[ind]==',':
                    label_each.append(int(string))
                    string = ''
                    ind+=1
                    if ind>len(contents)-1:
                        break
                elif contents[ind]==' ':
                    ind+=1
                    if ind>len(contents)-1:
                        break
                else:
                    string += contents[ind]
                    ind += 1
                    if contents[ind]==']':
                        label_each.append(int(string))
                        break
                    if ind>len(contents)-1:
                        break
            if len(label_each)>0:
                label.append(label_each)
                label_each = []
    for item in label:
        activity_list_box = []
        for i in item:
            activity_list_box.append(label_map[i])
        activity_list.append(activity_list_box)
    return activity_list
          
async def json_publish(primary):    
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    JSONEncoder = json.dumps(primary)
    json_encoded = JSONEncoder.encode()
    Subject = "service.activity"
    Stream_name = "service"
    await js.add_stream(name= Stream_name, subjects=[Subject])
    ack = await js.publish(Subject, json_encoded)
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")

async def batch_save(device_data, file_id):
    # BatchId = generate(size= 8)
    device_id = device_data[0]
    device_urn = device_data[1]
    timestampp = device_data[2]
    batchid = device_data[3] 
    
    global avg_Batchcount_person, avg_Batchcount_vehicel,avg_Batchcount_elephant, track_person,track_vehicle,track_elephant,detect_count,detect_img_cid,track_dir,track_type,batch_person_id

    video_name = path + '/' + str(device_id) +'/Nats_video'+str(device_id)+'-'+ str(file_id) +'.mp4'
    print(video_name)

    Process (target = await Activity(source=video_name,device_id=device_id,source_1=video_name)).start() 

    ct = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f') # ct stores current time
    timestamp = str(ct)
    activity_list = await BatchJson(source="classes.txt")
    metapeople ={
                    "type":ast.literal_eval(track_type),
                    "track":ast.literal_eval(track_person),
                    "id":ast.literal_eval(batch_person_id),
                    "activity":(activity_list),
                    "detect_time": (timestampp)
                    }
    
    metaVehicle = {
                    "type":ast.literal_eval(track_type),
                    "track":ast.literal_eval(track_vehicle),
                    "id":null,
                    "activity":null
                    }
    metaElephant = {
                    "track":ast.literal_eval(track_elephant),
                    "count":int(avg_Batchcount_elephant)
                    }
    metaObj = {
                "people":metapeople,
                "vehicle":metaVehicle,
                "elephant":metaElephant
            }
    
    metaBatch = {
        "detect": ast.literal_eval(detect_count),
        "count": {"people_count":int(avg_Batchcount_person),
                    "vehicle_count":int(avg_Batchcount_vehicel)} ,
        "object":metaObj,
        "cid":detect_img_cid
    }
    
    primary = { "pipeline": "activity",
                "deviceid": "uuid:eaadf637-a191-4ae7-8156-07433934718b",
                "batchid":(batchid), 
                "timestamp":(timestampp),
                "geo": {"latitude":'12.913632983105556',
                        "longitude":'77.58994246818435'}, 
                "metaData": metaBatch}
    print(primary , "Json final ")
    Process(target= await json_publish(primary=primary)).start()
    detect_count = []
    # avg_Batchcount_person = []
    # avg_Batchcount_vehicel = []
    track_person = []
    track_vehicle = []
    track_elephant = []
    activity_list.clear()
    # detect_img_cid = []
    track_type = []
    batch_person_id = []
    # avg_Batchcount_elephant = []
    os.remove("classes.txt")
    shutil.rmtree(track_dir)
    gc.collect()
    torch.cuda.empty_cache()

async def gst_data(file_id , device_data):
    
    global count 
    count = count + 1
    sem = asyncio.Semaphore(1)
    device_id = device_data[0]
    device_urn = device_data[1]
    timestampp = device_data[2]
    device_data.append(count)
    await sem.acquire()
    try:
        if device_id not in devicesUnique:
            t = Process(target= await batch_save(device_data=device_data ,file_id=file_id))
            t.start()
            processes.append(t)
            devicesUnique.append(device_id)
        else:
            ind = devicesUnique.index(device_id)
            t = processes[ind]
            Process(name = t.name, target= await batch_save(device_data=device_data ,file_id=file_id))
    
    except TypeError as e:
        print(TypeError," gstreamer error 121 >> ", e)
        
    finally:
        print("done with work ")
        sem.release()

    logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
    logging.debug("Debug logging test...")
    logging.info("Program is working as expected")
    logging.warning("Warning, the program may not function properly")
    logging.error("The program encountered an error")
    logging.critical("The program crashed")
    
async def device_snap_pub(device_id, urn, gif_cid, time_stamp):
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    device_data = {
        "deviceId": device_id,
        "urn": urn,
        "timestamp": time_stamp,
        "thumbnail": gif_cid,
        # "uri": "https://hls.ckdr.co.in/live/stream1/1.m3u8",
        # "status": "active" 
    }
    JSONEncoder = json.dumps(device_data)
    json_encoded = JSONEncoder.encode()
    print(json_encoded)
    print("Json encoded")
    Subject = "service.device_thumbnail"
    Stream_name = "service"
    # await js.add_stream(name= Stream_name, subjects=[Subject])
    ack = await js.publish(Subject, json_encoded)
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")
    
async def gif_creation(device_id, urn, img_arr, timestamp):
    global image_count, gif_path
    
    path = gif_path + '/' + str(timestamp).replace(' ','') + '.gif'
    
    image_count += 1
    if (image_count < 30):
        frames.append(img_arr)
    elif (image_count >= 30):
        print(timestamp)
        print("Images added: ", len(frames))
        print("Saving GIF file")
        with imageio.get_writer(path, mode="I") as writer:
            for idx, frame in enumerate(frames):
                print("Adding frame to GIF file: ", idx + 1)
                writer.append_data(frame)
                
        print("PATH:", path)
        command = 'ipfs --api=/ip4/216.48.181.154/tcp/5001 add {file_path} -Q'.format(file_path=path)
        gif_cid = sp.getoutput(command)
        print(gif_cid)
        await device_snap_pub(device_id = device_id, urn=urn, gif_cid = gif_cid, time_stamp = timestamp)
        os.remove(path)
        frames.clear()
        image_count = 0
    
async def hls_stream(gst_stream_data):
    
    global gst_str
    
    camID = gst_stream_data[0]
    urn = gst_stream_data[1]
    rtsp_URL = gst_stream_data[2]
    camera_type = gst_stream_data[3]
    
    print("Entering HLS Stream")
    
    # filename for hls
    video_name_hls1 = hls_path + '/' + str(camID)
    if not os.path.exists(video_name_hls1):
        os.makedirs(video_name_hls1, exist_ok=True)
        
    print(video_name_hls1)
    
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # print(caps.get_structure(0).get_value('height'), caps.get_structure(0).get_value('width'))
            
        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
            caps.get_structure(0).get_value('width')),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)

        return arr

    def new_buffer(sink, data):
        global image_arr
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        arr = gst_to_opencv(sample)
        rgb_frame = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        datetime_ist = str(datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f'))
        asyncio.run(gif_creation(device_id=camID, urn=urn, img_arr=rgb_frame, timestamp=datetime_ist))     
        return Gst.FlowReturn.OK
    
    # rtspsrc location={location} protocols="tcp" name={device_id} ! rtph264depay name=depay-{device_id} ! h264parse config_interval=-1 name=parse-{device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! video/x-raw,format=YUY2 ! appsink name=sink-{device_id}
    
    try:
        if((camera_type.lower()) == 'h264'):
            pipeline = Gst.parse_launch('rtspsrc name=m_rtspsrc_{ID} !  rtph264depay name=m_depay_{ID} ! tee name=t t. ! queue ! h264parse config_interval=-1 name=m_parse_{ID} ! decodebin name=m_decode_{ID} ! appsink name=m_sink_{ID}'.format(ID = camID))
        if((camera_type.lower()) == 'h265'):
            pipeline = Gst.parse_launch('rtspsrc name=m_rtspsrc_{ID} !  rtph265depay name=m_depay_{ID} ! tee name=t t. ! queue ! h265parse config_interval=-1 name=m_parse_{ID} ! decodebin name=m_decode_{ID} ! appsink name=m_sink_{ID}'.format(ID = camID))

        # source params
        source = pipeline.get_by_name('m_rtspsrc_{ID}'.format(ID = camID))
        source.set_property('latency', 30)
        source.set_property('location', rtsp_URL)
        source.set_property('protocols', 'tcp')
        source.set_property('drop-on-latency', 'true')

        # depay params
        depay = pipeline.get_by_name('m_depay_{ID}'.format(ID = camID))
        
        parse = pipeline.get_by_name('m_parse_{ID}'.format(ID = camID))
        
        decode = pipeline.get_by_name('m_decode_{ID}'.format(ID = camID))
        
        # convert = pipeline.get_by_name('m_convert_{ID}'.format(ID = camID))
        
        # encode = pipeline.get_by_name('m_enc_{ID}'.format(ID = camID))
        
        # mux params
        # mux = pipeline.get_by_name('m_mux_{ID}'.format(ID = camID))

        # sink params
        sink = pipeline.get_by_name('m_sink_{ID}'.format(ID = camID))
        # sink_1 = pipeline.get_by_name('m_sink1_{ID}'.format(ID = camID))
        
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", new_buffer, camID)

        # Location of the playlist to write
        # sink_1.set_property('playlist-root', 'https://hls.ckdr.co.in/live/stream{device_id}'.format(device_id = camID))
        # # Location of the playlist to write
        # sink_1.set_property('playlist-location', '{file_path}/{file_name}.m3u8'.format(file_path = video_name_hls1, file_name = camID))
        # # Location of the file to write
        # sink_1.set_property('location', '{file_path}/segment.%01d.ts'.format(file_path = video_name_hls1))
        # # The target duration in seconds of a segment/file. (0 - disabled, useful for management of segment duration by the streaming server)
        # sink_1.set_property('target-duration', 10)
        # # Length of HLS playlist. To allow players to conform to section 6.3.3 of the HLS specification, this should be at least 3. If set to 0, the playlist will be infinite.
        # sink_1.set_property('playlist-length', 3)
        # # Maximum number of files to keep on disk. Once the maximum is reached,old files start to be deleted to make room for new ones.
        # sink_1.set_property('max-files', 6)
        
        if not source or not sink or not pipeline or not depay or not parse or not decode:
            print("Not all elements could be created.")

        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.SUCCESS:
            print("Successfully set the pipeline to the playing state.")
            
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")
            
    except TypeError as e:
        print(TypeError," gstreamer hls streaming error >> ", e)

async def gst_stream(gst_stream_data):
     
    device_id = gst_stream_data[0]
    urn = gst_stream_data[1]
    location = gst_stream_data[2]
    device_type = gst_stream_data[3]
    
    def format_location_callback(mux, file_id, data):
        
        device_data = []
        global start_flag
        device_data.append(data)
        device_data.append(urn)
        device_data.append(timestampp)
        
        if file_id == 0:
            if start_flag:
                pass
            else:
                file_id = 4
                asyncio.run(gst_data(file_id , device_data))
        else:
            file_id = file_id - 1
            start_flag = False
            asyncio.run(gst_data(file_id , device_data))

        print(file_id, '-------', data)
        
    def frame_timestamp(identity, buffer, data):
        global timestampp
        timestampp =  datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f')

    try:
        # filename for mp4
        video_name1 = path + '/' + str(device_id)
        print(video_name1)
        if not os.path.exists(video_name1):
            os.makedirs(video_name1, exist_ok=True)
        video_name = video_name1 + '/Nats_video'+str(device_id)
        print(video_name)
    
        if((device_type.lower()) == "h264"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! identity name=ident-{device_id} ! rtph264depay name=depay-{device_id} ! h264parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videorate name=rate-{device_id} ! video/x-raw,framerate=25/1 ! x264enc name=enc-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-files=5 max-size-time=20000000000 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))
        elif((device_type.lower()) == "h265"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! identity name=ident-{device_id} ! rtph265depay name=depay-{device_id} ! h265parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videorate name=rate-{device_id} ! video/x-raw,framerate=25/1 ! x264enc name=enc-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-files=5 max-size-time=20000000000 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))
        elif((device_type.lower()) == "mp4"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! identity name=ident-{device_id} ! rtph264depay name=depay-{device_id} ! h264parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videorate name=rate-{device_id} ! video/x-raw,framerate=25/1 ! x264enc name=enc-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-files=5 max-size-time=20000000000 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))

        sink = pipeline.get_by_name('sink-{device_id}'.format(device_id=device_id))
        identity = pipeline.get_by_name('ident-{device_id}'.format(device_id=device_id))
    
        if not pipeline:
            print("Not all elements could be created.")

        
        identity.connect("handoff", frame_timestamp, device_id)
        sink.connect("format-location", format_location_callback, device_id)
        
        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.SUCCESS:
            print("Able to set the pipeline to the playing state.")
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")

    except TypeError as e:
        print(TypeError," gstreamer streaming error >> ", e)
        
async def cb(msg):
    try :
        print("entered callback")
        data = (msg.data)
        data  = data.decode()
        data = json.loads(data)
        
        if "urn" not in data:
            data["urn"] = "uuid:eaadf637-a191-4ae7-8156-07433934718b"
        
        gst_stream_data = [data["deviceId"], data["urn"], data["rtsp"], data["videoEncodingInformation"]]
        
        if (data):
            # p2 = Process(target = await gst_stream(gst_stream_data))
            # p2.start()
            
            p3 = Process(target = await hls_stream(gst_stream_data))
            p3.start()

        subject = msg.subject
        reply = msg.reply
        await nc_client.publish(msg.reply,b'Received!')
        print("Received a message on '{subject} {reply}': {data}".format(
            subject=subject, reply=reply, data=data))
        
    except TypeError as e:
        print(TypeError," Nats msg callback error >> ", e)

async def main():
    
    pipeline = Gst.parse_launch('fakesrc ! queue ! fakesink')
    
    # Start pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.SUCCESS:
        print("Able to set the pipeline to the playing state.")
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Unable to set the pipeline to the playing state.")
        
    # data = {
    #     "deviceId": "uuid:eaadf637-a191-4ae7-8156-07433934718b",
    #     "rtsp": "rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
    #     "videoEncodingInformation": "h.264"  
    # }
        
    # p3 = Process(target = await hls_stream(data))
    # p3.start()
    
    await nc_client.connect(servers=["nats://216.48.181.154:5222"])
    print("Nats Connected")
    
    await nc_client.subscribe("service.device_discovery", cb=cb)
    print("subscribed")
    
    # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)
    del pipeline
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
        
"""
Json Object For a Batch Video 
JsonObjectBatch= {ID , TimeStamp , {Data} } 
Data = {
    "person" : [ Device Id , [Re-Id] , [Frame TimeStamp] , [Lat , Lon], [Person_count] ,[Activity] ]
    "car":[ Device ID, [Lp Number] , [Frame TimeStamp] , [Lat , lon] ]
}  
Activity = [ "walking" , "standing" , "riding a bike" , "talking", "running", "climbing ladder"]
"""

"""
metapeople ={
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"face_id",
                    "activity":{"activities":activity_list , "boundaryCrossing":boundary}  
                    }
    
    metaVehicel = {
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"license_plate",
                    "activity":{"boundaryCrossing":boundary}
    }
    metaObj = {
                 "people":metapeople,
                 "vehicle":metaVehicel
               }
    
    metaBatch = {
        "Detect": "0: detection NO, 1: detection YES",
        "Count": {"people_count":str(avg_Batchcount),
                  "vehicle_count":str(avg_Batchcount)} ,
        "Object":metaObj
        
    }
    
    primary = { "deviceid":str(Device_id),
                "batchid":str(BatchId), 
                "timestamp":str(frame_timestamp), 
                "geo":str(Geo_location),
                "metaData": metaBatch}
    print(primary)
    
"""