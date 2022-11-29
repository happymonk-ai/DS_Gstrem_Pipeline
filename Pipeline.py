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

#Detection
from track import run

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

# face_detection
# import lmdb

path = "./Nats_output"

if os.path.exists(path) is False:
    os.mkdir(path)
    
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
avg_Batchcount_person =[]
avg_Batchcount_vehicel = []
activity_list= []
geo_locations = []
track_person = []
track_vehicle = []
batch_person_id = []

queue1 = Queue()
queue2 = Queue()
queue3 = Queue()
queue4 = Queue()
queue5 = Queue()

device = 'cuda' # or 'cpu'
video_model = slow_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)

# gstreamer
# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)
image_arr = None
device_types = ['', 'h.264', 'h.264', 'h.264', 'h.265', 'h.265', 'h.264']
load_dotenv()

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
    global avg_Batchcount_person

    # Create an id to label name mapping
    global count_video            
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
    # Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
    video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.5)
    
    encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(source)
    
    time_stamp_range = range(1,9) # time stamps in video for which clip is sampled. 
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
            print("Skipping clip no frames detected at time stamp: ", time_stamp)
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
        det = Process(target= run(source=vide_save_path, queue1=queue1,queue2=queue2,queue3=queue3,queue4=queue4,queue5=queue5))
        det.start()
        avg_Batchcount_person = queue1.get()
        avg_Batchcount_vehicel= queue2.get()
        detect_count= queue3.get()
        track_person = queue4.get()
        track_vehicle = queue5.get()
        
    except IndexError:
        print("No Activity")
        activity_list.append("No Activity")
        open('classes.txt','w')
        await asyncio.sleep(1)
        det = Process(target= run(source=source_1, queue1=queue1,queue2=queue2,queue3=queue3,queue4=queue4,queue5=queue5))
        det.start()
        avg_Batchcount_person = queue1.get()
        avg_Batchcount_vehicel = queue2.get()
        detect_count= queue3.get()
        track_person = queue4.get()
        track_vehicle = queue5.get()
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
    Subject = "model.activity_v1"
    Stream_name = "Testing_json"
    await js.add_stream(name= Stream_name, subjects=[Subject])
    ack = await js.publish(Subject, json_encoded)
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")

async def Video_creating(path, device_id):
    image_folder = path
    video_name = path+'/Nats_video'+str(device_id)+'.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc , 1, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()
    Process (target = await Activity(source=video_name,device_id=device_id,source_1=video_name)).start() 
    await asyncio.sleep(1)
    

async def batch_save(device_id,time_stamp):
    global avg_Batchcount_person
    BatchId = generate(size= 8)
    count_batch = 0
    for msg in dict_frame[device_id]:
        try :
            im = np.ndarray(
                    (512,
                    512),
                    buffer=np.array(msg),
                    dtype=np.uint8) 
            im = Image.fromarray(im)
            device_path = os.path.join(path,str(device_id))
            if os.path.exists(device_path) is False:
                os.mkdir(device_path)
            im.save(device_path+"/"+str(count_batch)+".jpeg")
            await asyncio.sleep(1)
        except TypeError as e:
            print(TypeError," gstreamer error 64 >> ",e,"Device Id",device_id)
        count_batch += 1
    Process(target=await Video_creating(path=device_path, device_id=device_id)).start()
    await asyncio.sleep(1)
    if count_batch >= 10 :
        pattern = device_path+"/**/*.jpeg"
        for item in glob.iglob(pattern, recursive=True):
            os.remove(item) 
    activity_list = await BatchJson(source="classes.txt")
    metapeople ={
                    "type":str(track_type),
                    "track":str(track_person),
                    "id":batch_person_id,
                    "activity":{"activities":activity_list}  
                    }
    
    metaVehicle = {
                    "type":str(track_type),
                    "track":str(track_vehicle),
    }
    metaObj = {
                "people":metapeople,
                "vehicle":metaVehicle
            }
    
    metaBatch = {
        "Detect": str(detect_count),
        "Count": {"people_count":str(avg_Batchcount_person),
                    "vehicle_count":str(avg_Batchcount_vehicel)} ,
                "Object":metaObj
    }
    
    primary = { "deviceid":str(device_id),
                "batchid":str(BatchId), 
                "timestamp":str(time_stamp), 
                "metaData": metaBatch}
    print(primary)
    Process(target= await json_publish(primary=primary)).start()
    dict_frame[device_id].clear()
    count_frame[device_id] = 0 
    detect_count.clear()
    avg_Batchcount_person = []
    avg_Batchcount_vehicel.clear()
    track_person.clear()
    track_vehicle.clear()
    person_count.clear()
    vehicle_count.clear()
    activity_list.clear()
    os.remove("classes.txt")
    gc.collect()
    torch.cuda.empty_cache()

                
async def stream_thread(device_id , frame_byte,timestamp) :
    if len(dict_frame) == 0 :
        dict_frame[device_id] = list(frame_byte)
        count_frame[device_id] = 1 
    else:
        if device_id in list(dict_frame.keys()):
            dict_frame[device_id].append(list(frame_byte))
            count_frame[device_id] += 1
            if count_frame[device_id] % 10 == 0 :
                Process(target = await batch_save(device_id=device_id,time_stamp=timestamp)).start()
                await asyncio.sleep(1)
        else:
            dict_frame[device_id] = list(frame_byte)
            count_frame[device_id] = 1
    print(count_frame, "count frame ", threading.get_ident(),"Threading Id" ,device_id ,"Device id")
    await asyncio.sleep(1)


async def gst_data(device_id ,frame_byte, timestamp):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try:
        if device_id not in devicesUnique:
            t = Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp))
            t.start()
            processes.append(t)
            devicesUnique.append(device_id)
        else:
            ind = devicesUnique.index(device_id)
            t = processes[ind]
            Process(name = t.name, target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp))
    
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

async def gst_stream(device_id, location, device_type):
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
            
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
        timestamp = buffer.pts
        arr = gst_to_opencv(sample)
        resized = cv2.resize(arr, (512, 512))
        asyncio.run(gst_data(device_id=data ,frame_byte=resized, timestamp=timestamp))     
        return Gst.FlowReturn.OK

    try:
        # pipeline = Gst.parse_launch('filesrc location={location} name={device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 1024, height = 1024 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))
        if(device_type == "h.264"):
            pipeline = Gst.parse_launch('rtspsrc location={location} name={device_id} ! queue max-size-buffers=2 ! rtph264depay name=depay-{device_id} ! h264parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 512, height = 512 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))
        elif(device_type == "h.265"):
            pipeline = Gst.parse_launch('rtspsrc location={location} name={device_id} ! queue max-size-buffers=2 ! rtph265depay name=depay-{device_id} ! h265parse name=parse-{device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 512, height = 512 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))

        sink = pipeline.get_by_name('sink-{device_id}'.format(device_id=device_id))

        if not pipeline:
            print("Not all elements could be created.")
        
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", new_buffer, device_id)
        
        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")

    except TypeError as e:
        print(TypeError," gstreamer streaming error >> ", e)

def on_message(bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
    mtype = message.type
    """
        Gstreamer Message Types and how to parse
        https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
    """
    if mtype == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()

    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(("Error received from element %s: %s" % (
            message.src.get_name(), err)))
        print(("Debugging information: %s" % debug))
        loop.quit()

    elif mtype == Gst.MessageType.STATE_CHANGED:
        if isinstance(message.src, Gst.Pipeline):
            old_state, new_state, pending_state = message.parse_state_changed()
            print(("Pipeline state changed from %s to %s." %
            (old_state.value_nick, new_state.value_nick)))
    return True

async def main():
    
    pipeline = Gst.parse_launch('fakesrc ! queue ! fakesink')

    # Init GObject loop to handle Gstreamer Bus Events
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    # allow bus to emit messages to main thread
    bus.add_signal_watch()

    # Add handler to specific signal
    bus.connect("message", on_message, loop)

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    for i in range(1, 7):
        stream_url = os.getenv('RTSP_URL_{id}'.format(id=i))
        t = Process(target= await gst_stream(device_id=i ,location=stream_url, device_type=device_types[i]))
        t.start()
    
    try:
        loop.run()
    except Exception:
        traceback.print_exc()
        loop.quit()

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