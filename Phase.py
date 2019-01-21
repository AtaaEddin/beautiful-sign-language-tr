import time
import json
import os

import numpy as np
from datagenerator import VideoClasses
import cv2

from timer import Timer
from frame import image_crop, frames_show, images_normalize, frames_downsample, images_rescale,video_length
from opticalflow import OpticalFlow, frames2flows, flow2colorimage, flows2colorimages, unittest_fromfile
from predict import get_predicts,load_Models,i3d_LSTM_prediction
from multiprocessing.pool import ThreadPool
from model_i3d import load_model_without_topLayer

import math
import keras
import queue
from queue import Queue
import requests
from urllib.parse import urlencode
from threading import Lock
import threading
import multiprocessing 
#import asyncio
import subprocess
from functools import reduce

ids_fpaths_q = Queue()
id_and_prediction_q = Queue()


def work(url, sec_sleep=2):
	fpath = ""
	global id_and_prediction_q
	global ids_fpaths_q
	print(f"SERVER will check for new entry every {sec_sleep} sec")
	print("SERVER listening....")
	lock = False
	id_and_prediction = None
	while 1:
		#time.sleep(sec_sleep)
		#print("check ediyorum..."+url)
		id_and_path = get_vid_dirs(url)
		if id_and_path is not None and not lock:
			vid_id,vid_path = id_and_path
			print("="*50)
			print(f"SERVER: [GET]: FROM 'videos' table - Get raw (vid_id={vid_id}, vid_path={vid_path})")
			
			if not ids_fpaths_q.full():
				ids_fpaths_q.put(id_and_path)
			
			lock = True
		
		if not id_and_prediction_q.empty():
			id_and_prediction = id_and_prediction_q.get()

		if id_and_prediction is not None:
			#queue.task_done()
			
			#id_and_prediction[1].replace("-"," ")
			print(id_and_prediction[1])
			new_id_and_prediction = []
			for item in id_and_prediction[1]:
				for key,val in item.items():
					new_id_and_prediction.append({key.replace("-"," ") : val})

			post_vid_results(url, id_and_prediction[0], new_id_and_prediction)
			print(f"SERVER: [POST]: TO 'keywords' table - Insert raw (id={id_and_prediction[0]}, keywords:{id_and_prediction[1]})")
			print("="*50)
			#time.sleep(2)
			lock = False
			id_and_prediction = None

	print("done")

def get_vid_dirs(url):
	mydata = {"operationtype" : "get_videos_path"}
	headers = {'format': "application/x-www-form-urlencoded"}

	resp = requests.post(url, data=mydata, headers=headers)

	resp = resp.content.decode()
	resp = json.loads(resp)

	if resp["message"] == "Data Yok":
		return None
	else:
		vid_info = resp["video_list"][0]
		
		return (vid_info['id'],vid_info["path"])

def vid_info(cap):
	info = []
	info.append(f"frameCount = {(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
	info.append(f"frameWidth = {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
	info.append(f"frameHeight = {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
	info.append(f"FPS = {cap.get(cv2.CAP_PROP_FPS)}")
	"""
	cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
	info.append(f"duration: {cap.get(cv2.CAP_PROP_POS_MSEC)}")
	cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
	"""
	return info

def getLength(filename):
  result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines()]

def post_vid_results(url, vid_id:int, keywords:list):
	mydata = {"operationtype" : "update_videos", "id":vid_id}
	mydata["keywords"] = json.dumps(keywords)
	
	resp = requests.post(url, data=mydata)
	

	#print(resp.text)

def image_normalize(rgbFrames, nFrames):
	rgbFrames = frames_downsample(rgbFrames, 40)
	rgbFrames = images_rescale(rgbFrames)
	return rgbFrames

def vid2frames(vid):
	oOpticalFlow = OpticalFlow(bThirdChannel = False)
	tuRectangle = (224, 224)
	success, frame = vid.read()
	rgbFrames = []
	oflowFrames = []
	frames = 0
	fails = 10
	while fails > 0 :

		if success:
			frames += 1

			frame = cv2.flip(frame, 1)
			frame = cv2.resize(frame, tuRectangle, interpolation =cv2.INTER_AREA)

			rgbFrames.append(frame)
			
			oflow = oOpticalFlow.next(frame)
			oflowFrames.append(oflow)

		else:
			fails -= 1

		success, frame = vid.read()

#	success, frame = vid.read()	
#		if not success:
#			print("[warrning]: some video frames are corrupted.")
	
	#print(f"duration: {vid.get(cv2.CAP_PROP_POS_MSEC)}")
	rgbFrames = images_rescale(np.array(rgbFrames))
	#oflowFrames = frames_downsample(np.array(oflowFrames), 40)

	#print(rgbFrames.shape)
	#print(oflowFrames.shape)

	print("frames_count:", frames)

	return rgbFrames, oflowFrames, frames


#SLD(sign-language-detection) 
def start_SLD_server(host_url, host_root, use_Edited_Model, i3d_models, LSTM_model, csvFile_dir, nTop= 3):
	global id_and_prediction
	global ids_fpaths_q
	labels = VideoClasses(csvFile_dir)

	rgb_model,oflow_model,lstmModel = load_Models(i3d_models, LSTM_model, use_Edited_Model)

	threading.Thread(target=work, args=(host_url, 2)).start()

	result = None
	processed_vid = []
	while True:
		
		if not ids_fpaths_q.empty():
			result = ids_fpaths_q.get()
		
		if result is not None:
			vid_id,vid_path = result
			processed_vid.append(vid_id)
			
			if not os.path.exists(host_root + vid_path):
				raise ValueError("[ERROR]: Incorrect pathing to the videos - (check videos dirctories).")
			
			vid = cv2.VideoCapture(host_root + vid_path)
			
			rgbs,oflows,frames_count = vid2frames(vid)


			results = preds(rgbs,oflows,frames_count,labels,use_Edited_Model,nTop,lstmModel,rgb_model,oflow_model,40,10,40)
			
			if len(results) > 0:
				predictions = Phase(results)
			else:
				predictions = [{'Unknown': 0.}, {'Unknown': 0.}, {'Unknown': 0.}]
			print("My results:", results)
			"""
			if use_Edited_Model:
				predictions,_ = i3d_LSTM_prediction(rgbs, oflows, labels, lstmModel, rgb_model, oflow_model, nTop=3)
			else:
				predictions,_ = get_predicts(rgbs, oflows, labels, oflow_model, rgb_model, nTop=3)
			"""
			#print("predictions:",predictions)
			if not id_and_prediction_q.full() and vid_id in processed_vid:
				id_and_prediction_q.put((vid_id,predictions))
				processed_vid.remove(vid_id)
			result = None


if __name__ == '__main__':

	i3d_models = {"oflow" : "./model/10_turkish_class/20181129-1002-chalearn035-oflow-i3d-entire-best.h5",#"./model/20181011-1229-chalearn249-oflow-i3d-entire-best.h5",
	"rgb" : "./model/10_turkish_class/20181129-0800-chalearn035-rgb-i3d-entire-best_acc_98.h5"}#}#"./model/20181015-1456-chalearn249-rgb-i3d-entire-best.h5"}
	LSTM = "./model/10_turkish_class/-1543523779-last_best_so_far_LSTM_FC_256.h5"

	start_SLD_server(host_url="http://localhost/combine/webservices.php",
					host_root="C:/wamp64/www/combine/",
					use_Edited_Model = True,
					i3d_models=i3d_models,
					LSTM_model=LSTM,
					csvFile_dir="./turkish_classes_edited.csv")

