import argparse
import asyncio

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from math import ceil
from PIL import Image
from multiprocessing import Process, Queue
import subprocess as sp
import glob
from nanoid import generate
from io import BytesIO

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from matplotlib import gridspec

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

from keras.applications.resnet import ResNet50 
#from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet import preprocess_input 
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from collections import Counter

#face_detection
import lmdb
# import face_lmdb
import json
import face_recognition 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'Detection') not in sys.path:
    sys.path.append(str(ROOT / 'Detection'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from Detection.models.common import DetectMultiBackend
from Detection.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from Detection.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from Detection.utils.torch_utils import select_device, time_sync
from Detection.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

 # Load model
device_track=''
devices = select_device(device_track)
model = DetectMultiBackend(WEIGHTS / '27Sep_2022.pt', device=devices, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size

TOLERANCE = 0.62
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'svm'
person_count = []
vehicle_count = []
elephant_count = []
avg_Batchcount_person =[]
avg_Batchcount_vehicel = []
avg_Batchcount_elephant = []
track_person = []
track_vehicle = []
track_elephant = []
batch_person_id = []
detect_count = []
detect_file = []
detect_img_cid = []
detect_image = {}
count_person =0
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
face_did_encoding_store = dict()
track_type = []

# #load lmdb
# env = lmdb.open('/home/nivetheni/DS_Gstrem_Pipeline/lmdb/face-detection.lmdb',
#                 max_dbs=10, map_size=int(100e9))

# # Now create subdbs for known and unknown people.
# known_db = env.open_db(b'known')
# unknown_db = env.open_db(b'unknown')

# async def lmdb_known():
#     # Iterate each DB to show the keys are sorted:
#     with env.begin() as txn:
#         list1 = list(txn.cursor(db=known_db))
        
#     db_count_whitelist = 0
#     for key, value in list1:
#         #fetch from lmdb
#         with env.begin() as txn:
#             re_image = txn.get(key, db=known_db)
#             # Deserialization
#             print("Decode JSON serialized NumPy array")
#             decodedArrays = json.loads(re_image)
            
#             finalNumpyArray = np.asarray(decodedArrays["array"], dtype="uint8")
            
#             # Load an image
#             # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
#             image = finalNumpyArray
#             ratio = np.amax(image) / 256
#             image = (image / ratio).astype('uint8')
            
#             # Get 128-dimension face encoding
#             # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
#             try :
#                 encoding = face_recognition.face_encodings(image)[0]
#             except IndexError as e  :
#                 print( "Error ", IndexError , e)
#                 continue
            
#             # Append encodings and name
#             known_whitelist_faces.append(encoding)
#             known_whitelist_id.append(key.decode())
#             db_count_whitelist += 1
            
#     print(db_count_whitelist, "total whitelist person")

# async def lmdb_unknown():
#     # Iterate each DB to show the keys are sorted:
#     with env.begin() as txn:
#         list2 = list(txn.cursor(db=unknown_db))
        
#     db_count_blacklist = 0
#     for key, value in list2:
#         #fetch from lmdb
#         with env.begin() as txn:
#             re_image = txn.get(key, db=unknown_db)
#             # Deserialization
#             print("Decode JSON serialized NumPy array")
#             decodedArrays = json.loads(re_image)
            
#             finalNumpyArray = np.asarray(decodedArrays["array"],dtype="uint8")
            
#             # Load an image
#             # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
#             image = finalNumpyArray
#             ratio = np.amax(image) / 256
#             image = (image / ratio).astype('uint8')
            
#             # Get 128-dimension face encoding
#             # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
#             try :
#                 encoding = face_recognition.face_encodings(image)[0]
#             except IndexError as e  :
#                 print( "Error ", IndexError , e)
#                 continue
            
#             # Append encodings and name
#             known_blacklist_faces.append(encoding)
#             known_blacklist_id.append(key.decode())
#             db_count_blacklist += 1
    
#     print(db_count_blacklist, "total blacklist person")

@torch.no_grad()
def run(
        source = ROOT,
        queue1 = Queue(),
        queue2 = Queue(),
        queue3 = Queue(),
        queue4 = Queue(),
        queue5 = Queue(),
        queue6 = Queue(),
        queue7 = Queue(),
        queue8 = Queue(),
        queue9 = Queue(),
        queue10 = Queue(),
        yolo_weights=WEIGHTS / '27Sep_2022.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'Nats_output/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, devices, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    # frame_count = 0
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(devices)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                # Print results
                for c in det[:, -1].unique():
                    global vehicle_count , license, detect_image, elephant_count
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == "Person" :
                        print(f"{n}","line 338")
                        person_count.append(int(f"{n}"))
                        print("person detected")
                    if names[int(c)] == "Vehicle":
                       vehicle_count.append(int(f"{n}"))
                       print("vehicel detected")
                    if names[int(c)] == "Elephant":
                       elephant_count.append(int(f"{n}"))
                       print("elephant detected")
                # for c in det[:,-1]:
                #     global personDid , count_person 
                #     # if frame_count % 10 == 0: 
                #     if names[int(c)]=="Face":
                #         print("person detected starting face detection code ")
                #         count_person += 1
                #         if count_person>0:
                #             np_bytes2 = BytesIO()
                #             np.save(np_bytes2, im0, allow_pickle=True)
                #             np_bytes2 = np_bytes2.getvalue()

                #             image = im0 # if im0 does not work, try with im1
                #             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                #             locations = face_recognition.face_locations(image, model=MODEL)

                #             encodings = face_recognition.face_encodings(image, locations)
                                
                #             print(f', found {len(encodings)} face(s)\n')
                                
                #             for face_encoding ,face_location in zip(encodings, locations):
                #                     print(np.shape(known_whitelist_faces), "known_whitelist_faces", np.shape(face_encoding),"face_encoding")
                #                     results_whitelist = face_recognition.compare_faces(known_whitelist_faces, face_encoding, TOLERANCE)
                #                     print(results_whitelist, "611")
                #                     if True in results_whitelist:
                #                         did = '00'+ str(known_whitelist_id[results_whitelist.index(True)])
                #                         print(did, "did 613")
                #                         batch_person_id.append(did)
                #                         track_type.append("00")
                #                         ct = datetime.datetime.now() # ct stores current time
                #                         timestamp.append(str(ct))
                #                         if did in face_did_encoding_store.keys():
                #                             face_did_encoding_store[did].append(face_encoding)
                #                             top_left = (face_location[3], face_location[0])
                #                             bottom_right = (face_location[1], face_location[2])
                #                             color = [0,255,0]
                #                             cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                             top_left = (face_location[3], face_location[2])
                #                             bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                             cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                             cv2.putText(im0, did , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                                
                #                         else:
                #                             face_did_encoding_store[did] = list(face_encoding)
                #                             top_left = (face_location[3], face_location[0])
                #                             bottom_right = (face_location[1], face_location[2])
                #                             color = [0,255,0]
                #                             cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                             top_left = (face_location[3], face_location[2])
                #                             bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                             cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                             cv2.putText(im0, did , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                                                
                #                     else:
                #                         # if face_encoding not in known_blacklist_faces:
                #                             # # Serialization
                #                             # numpyData = {"array": image}
                #                             # encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
                #                             # #push to blacklist lmdb
                #                             # person_name = bytearray(name[0]+ str(count), "utf-8")
                #                             # person_img = bytearray(encodedNumpyData, "utf-8")
                #                             # with env.begin(write=True) as txn:
                #                             #     txn.put(person_name, person_img, db=known_db)
                #                             known_blacklist_faces.append(face_encoding)
                #                         # else:
                #                             results_blacklist = face_recognition.compare_faces(known_blacklist_faces, face_encoding, TOLERANCE)
                #                             if True in results_blacklist:
                #                                 # did = '01'+ str(known_blacklist_id[results_blacklist.index(True)])
                #                                 did = '01' + 'blacklist'
                #                                 print("did 623", did)
                #                                 batch_person_id.append(did)
                #                                 track_type.append("01")
                #                                 ct = datetime.datetime.now() # ct stores current time
                #                                 timestamp.append(str(ct))
                #                                 if did in face_did_encoding_store.keys():
                #                                     face_did_encoding_store[did].append(face_encoding)
                #                                     top_left = (face_location[3], face_location[0])
                #                                     bottom_right = (face_location[1], face_location[2])
                #                                     color = [0,255,0]
                #                                     cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                                     top_left = (face_location[3], face_location[2])
                #                                     bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                                     cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                                     cv2.putText(im0, did , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                #                                 else:
                #                                     face_did_encoding_store[did] = list(face_encoding)
                #                                     top_left = (face_location[3], face_location[0])
                #                                     bottom_right = (face_location[1], face_location[2])
                #                                     color = [0,255,0]
                #                                     cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                                     top_left = (face_location[3], face_location[2])
                #                                     bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                                     cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                                     cv2.putText(im0, did , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                                                        
                #                             else:
                #                                 if len(face_did_encoding_store) == 0:
                #                                     did = '10'+ str(generate(size =4 ))
                #                                     print(did, "did 642")
                #                                     track_type.append("10")
                #                                     batch_person_id.append(did)
                #                                     face_did_encoding_store[did] = list(face_encoding)
                #                                     top_left = (face_location[3], face_location[0])
                #                                     bottom_right = (face_location[1], face_location[2])
                #                                     color = [0,255,0]
                #                                     cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                                     top_left = (face_location[3], face_location[2])
                #                                     bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                                     cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                                     cv2.putText(im0, did , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                                                    
                #                                 else:
                #                                     for key, value in face_did_encoding_store.items():
                #                                         print(key,"640")
                #                                         if key.startswith('10'):
                #                                             try :
                #                                                 print(type(value),"type vlaue")
                #                                                 print(np.shape(np.transpose(np.array(value))), "value 642" ,np.shape(value) ,"value orginal",np.shape(face_encoding), "face_encoding")
                #                                                 results_unknown = face_recognition.compare_faces(np.transpose(np.array(value)), face_encoding, TOLERANCE)
                #                                                 # results_unknown = face_recognition.compare_faces(np.array(value), face_encoding, TOLERANCE)
                #                                                 print(results_unknown,"635")
                #                                                 if True in results_unknown:
                #                                                     key_list = list(key)
                #                                                     key_list[1] = '1'
                #                                                     key = str(key_list)
                #                                                     print(key, "did 637")
                #                                                     batch_person_id.append(key)
                #                                                     track_type.append("11")
                #                                                     face_did_encoding_store[key].append(face_encoding)
                #                                                     top_left = (face_location[3], face_location[0])
                #                                                     bottom_right = (face_location[1], face_location[2])
                #                                                     color = [0,255,0]
                #                                                     cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                                                     top_left = (face_location[3], face_location[2])
                #                                                     bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                                                     cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                                                     cv2.putText(im0, key , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                                                                    
                #                                                 else:
                #                                                     did = '10'+ str(generate(size=4))
                #                                                     print(did, "did 642")
                #                                                     batch_person_id.append(did)
                #                                                     face_did_encoding_store[did] = list(face_encoding)
                #                                                     top_left = (face_location[3], face_location[0])
                #                                                     bottom_right = (face_location[1], face_location[2])
                #                                                     color = [0,255,0]
                #                                                     cv2.rectangle(im0, top_left, bottom_right, color, FRAME_THICKNESS)
                #                                                     top_left = (face_location[3], face_location[2])
                #                                                     bottom_right = (face_location[1]+50, face_location[2] + 22)
                #                                                     cv2.rectangle(im0, top_left, bottom_right, color, cv2.FILLED)
                #                                                     cv2.putText(im0, did , (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), FONT_THICKNESS)
                                                                
                #                                             except np.AxisError as e:
                #                                                 print(e,">> line 562")
                #                                                 continue
                                                        
                #                                 print(batch_person_id, "batch_person_id")

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                crop_img = save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                                detect_file_path = save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg'
                                if detect_file_path not in detect_file:
                                    detect_file.append(str(detect_file_path))
                                    detect_image = (set(detect_file))
                                
                                
                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

            else:
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
           
            # Save results (image with detections)
            if save_vid:
                # # if dataset.mode == 'image':
                image_path = (str(save_dir) + '/detect')
                if not os.path.exists(image_path):
                    os.makedirs(image_path, exist_ok=True)
                image_path1 = str(image_path) + '/detect.jpg'
                cv2.imwrite(image_path1, im0) # to save detected image

                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

            # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        
    #people Count
    sum_count = 0
    for x in person_count:
        sum_count += int(x)
    try :
        avg = ceil(sum_count/len(person_count))
        avg_Batchcount_person.append(str(avg))
    except ZeroDivisionError:
        avg_Batchcount_person.append("0")
        print("No person found ")
        
    for iten in avg_Batchcount_person:
        for i in range(int(iten[0])):
            track_person.append(1)
        
    sum_count = 0
    for x in vehicle_count:
        sum_count += int(x)
    try :
        avg = ceil(sum_count/len(vehicle_count))
        avg_Batchcount_vehicel.append(str(avg))
    except ZeroDivisionError:
        avg_Batchcount_vehicel.append("0")
        print("No Vehicle found ")
    
    for iten in avg_Batchcount_vehicel:
        for i in range(int(iten[0])):
            track_vehicle.append(1)

    sum_count = 0
    for x in elephant_count:
        sum_count += int(x)
    try :
        avg = ceil(sum_count/len(elephant_count))
        avg_Batchcount_elephant.append(str(avg))
    except ZeroDivisionError:
        avg_Batchcount_elephant.append("0")
        track_elephant.append(0)
        print("No Elephant found ")
    
    for iten in avg_Batchcount_elephant:
        for i in range(int(iten[0])):
            track_elephant.append(1)
        
    if len(person_count) > 0 or len(vehicle_count) > 0 or len(elephant_count) > 0 :
        detect_count.append(1)
    else:
        detect_count.append(0)
        
    # publish detected image to ipfs
    command = 'ipfs --api=/ip4/216.48.181.154/tcp/5001 add {file_path} -Q'.format(file_path=image_path1)
    output = sp.getoutput(command)
    detect_img_cid.append(output)


    queue1.put(str(avg_Batchcount_person))
    queue2.put(str(avg_Batchcount_vehicel))
    queue3.put(str(detect_count))
    queue4.put(str(track_person))
    queue5.put(str(track_vehicle))
    queue6.put(str(detect_img_cid))
    queue7.put(str(save_dir))
    queue8.put(str(track_type))
    queue9.put(str(batch_person_id))
    queue10.put(str(track_elephant))

    person_count.clear()
    vehicle_count.clear()
    avg_Batchcount_person.clear()
    avg_Batchcount_vehicel.clear()
    detect_count.clear()
    track_person.clear()
    track_vehicle.clear()
    track_elephant.clear()
    detect_file.clear()
    detect_img_cid.clear()
    track_type.clear()
    batch_person_id.clear()
    

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    run(source="gray_scale.mp4")
