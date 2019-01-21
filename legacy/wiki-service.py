import time
import json
import os

import numpy as np
from datagenerator import VideoClasses
import cv2
from threading import Timer
from frame import image_crop, frames_show, images_normalize, frames_downsample, images_rescale
from opticalflow import OpticalFlow, frames2flows, flow2colorimage, flows2colorimages, unittest_fromfile
from predict import get_predicts
from multiprocessing.pool import ThreadPool

import math
import keras
import queue
from queue import Queue
import requests
from urllib.parse import urlencode
from threading import Lock
import threading
import multiprocessing 
import asyncio

ids_fpaths_q = Queue(maxsize=1)
id_and_prediction_q = Queue(maxsize=1)
lock = False
waitForData = False
currentObj = None
currentResultObj = None

def work(url, sec_sleep=2):
	fpath = ""
	global id_and_prediction_q
	global ids_fpaths_q
	print(f"SERVER will check for new entry every {sec_sleep} sec")
	print("SERVER listening....")
	lock = False
	id_and_prediction = None
	while True:
		time.sleep(sec_sleep)
		print("check ediyorum..."+url)
		id_and_path = get_vid_dirs(url)
		if id_and_path is not None and not lock:
			vid_id,vid_path = id_and_path
			print("="*50)
			print(f"SERVER: [GET]: FROM 'videos' table - Get raw (vid_id={vid_id}, vid_path={vid_path})")
			
			try:
				ids_fpaths_q.put(id_and_path)
			except queue.Full:
				continue
			lock = True
		try:
			id_and_prediction = id_and_prediction_q.get()
		except queue.Empty:
			continue
		if id_and_prediction is not None:
			#queue.task_done()
			post_vid_results(url, id_and_prediction[0], id_and_prediction[1])
			print(f"SERVER: [POST]: TO 'keywords' table - Insert raw (id={id_and_prediction[0]}, keywords:{id_and_prediction[1]})")
			print("="*50)
			time.sleep(5)
			lock = False

def get_vid_dirs(url):
	print("="*50)
	print(ids_fpaths_q.qsize())
	if ids_fpaths_q.full():
		print("Awk! Queue is full! Lets look results!")
		print("Where is my words? Give me my words! Get it! Right now! Lets check result queue!")
		if id_and_prediction_q.empty():
			print("Sorry dude. Your data has not ready yet")
			r = Timer(2.0, get_vid_dirs, (url,))
			r.start()
			return

		id_and_prediction = id_and_prediction_q.get()
		print("Your data is ready! Here are:",id_and_prediction[0],id_and_prediction[1])
		#queue.task_done()
		print("I'm saving your data to db. Be cool my friend!")
		post_vid_results(url, id_and_prediction[0], id_and_prediction[1])
		print("="*50)
		r = Timer(2.0, get_vid_dirs, (url,))
		r.start()

	else:
		response = makeRequest(url)
		if response is not None:
			vid_id,vid_path = response
			currentObj = (vid_id,vid_path)
			print("I put data to waiting list! Be cool...",vid_id,vid_path)
			while True:
				currentObj
			r = Timer(2.0, get_vid_dirs, (url,))
			r.start()



def makeRequest(url):

	print("I will make a request for new data!")
	mydata = {"operationtype" : "get_videos_path"}
	headers = {'format': "application/x-www-form-urlencoded"}
	
	resp = requests.post(url, data=mydata, headers=headers)
	resp = resp.content.decode()
	resp = json.loads(resp)

	print("I took response",resp)
	if resp["message"] == "Data Yok":
		print("Data is empty. So I will try after 2 seconds!")
		return None;
	else:
		vid_info = resp["video_list"][0]
		return (vid_info['id'],vid_info["path"])

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
	while success:

		if not success:
			print("[warrning]: some video frames are corrupted.")

		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, tuRectangle, interpolation =cv2.INTER_AREA)
		rgbFrames.append(frame)
		
		oflow = oOpticalFlow.next(frame)
		oflowFrames.append(oflow)

		success, frame = vid.read()
	
	rgbFrames = image_normalize(np.array(rgbFrames), 40)
	oflowFrames = frames_downsample(np.array(oflowFrames), 40)

	#print(rgbFrames.shape)
	#print(oflowFrames.shape)

	return rgbFrames, oflowFrames




#SLD(sign-language-detection) 
def start_SLD_server(host_url, host_root, i3d_models, csvFile_dir, nTop= 3):
	global id_and_prediction
	global ids_fpaths_q
	labels = VideoClasses(csvFile_dir)
	
	rgb_model = None
	oflow_model = None
	if i3d_models["rgb"] is not None:
		rgb_model = keras.models.load_model(i3d_models["rgb"])
	if i3d_models["oflow"] is not None:
		oflow_model = keras.models.load_model(i3d_models["oflow"])

	#pool = multiprocessing.Pool(processes_num)
	#m = multiprocessing.Manager()
	#ids_fpaths_q = m.Queue()
	#id_and_prediction_q = m.Queue()
	#pool.apply_async(work, (host_url, ids_fpaths_q, id_and_prediction_q, 2))
	threading.Thread(target=get_vid_dirs, args=(host_url,)).start()

	

	result = None
	while True:
		result = currentObj
		if result is not None:
			vid_id,vid_path = result
			if not os.path.exists(host_root + vid_path):
				raise ValueError("[ERROR]: Incorrect pathing to the videos - (check videos dirctories).")
			vid = cv2.VideoCapture(host_root + vid_path)
			rgbs,oflows = vid2frames(vid)
			predictions,_ = get_predicts(rgbs, oflows, labels, oflow_model, rgb_model, nTop=3)
			#print("predictions:",predictions)
			currentResultObj = (vid_id,predictions)


if __name__ == '__main__':

	i3d_models = {"oflow" : "./model/35class/20181023-0930-chalearn035-oflow-i3d-entire-best_downloaded.h5",#"./model/20181011-1229-chalearn249-oflow-i3d-entire-best.h5",
	"rgb" : None}#"./model/35class/20181023-1505-chalearn035-rgb-i3d-entire-best_download.h5"}#"./model/20181015-1456-chalearn249-rgb-i3d-entire-best.h5"}

	start_SLD_server(host_url="http://localhost/combine/webservices.php",
					host_root="C:/wamp64/www/combine/",
					i3d_models=i3d_models,
					csvFile_dir="./35class.csv")
	#get_vid_dirs("http://localhost/haso/webservices.php")
	#post_vid_results("http://localhost/haso/webservices.php", 11, [{"anlamak": 90},{"gelmek" : 70},{"gitmek" : 20}])
