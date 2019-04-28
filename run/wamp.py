import time
import json
import os
import sys 
sys.path.insert(0,"./utils")
sys.path.insert(0,"./models")
import numpy as np


from queue import Queue
import requests
import threading
from utils.util import handler,load_models

#from urllib.parse import urlencode
#import math
#from timer import Timer
#import queue
#from threading import Lock
#import multiprocessing 
#import asyncio

ids_fpaths_q = Queue()
id_and_prediction_q = Queue()


def work(url, sec_sleep=2):
	fpath = ""
	global id_and_prediction_q
	global ids_fpaths_q
	#print(f"SERVER will check for new entry every {sec_sleep} sec")
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
			post_vid_results(url, id_and_prediction[0], id_and_prediction[1])
			print(f"SERVER: [POST]: TO 'keywords' table - Insert raw (id={id_and_prediction[0]}, keywords:{id_and_prediction[1]})")
			print("="*50)
			#time.sleep(2)
			lock = False
			id_and_prediction = None

	print("done")

def get_vid_dirs(url):
	mydata = {"operationtype" : "get_videos_path"}
	headers = {'format': "application/x-www-form-urlencoded"}
	resp = None
	try:
		resp = requests.post(url, data=mydata, headers=headers)
	except requests.exceptions.RequestException as e:
		print(e)
	except requests.exceptions.HTTPError as err:
		print(err)

	if resp == None:
		return None
	resp = resp.content.decode()
	try:
		resp = json.loads(resp)
	except Exception as e:
		print(e,"\nEither no data or worng formated data received.")

	if resp["message"] == "Data Yok":
		return None
	else:
		vid_info = resp["video_list"][0]
		
		return (vid_info['id'],vid_info["path"])


def post_vid_results(url, vid_id:int, keywords:list):
	mydata = {"operationtype" : "update_videos", "id":vid_id}
	mydata["keywords"] = json.dumps(keywords)
	print("post_vid_results called")
	resp = requests.post(url, data=mydata)
	

	#print(resp.text)



#SLD(sign-language-detection) 

"""
run_server(php_webservice,
			wamp_folder,
			use_rgb,
			use_oflow,
			use_lstm,
			on_cpu,
			on_gpu,
			pred_type,
			nTop
			)

"""

# [TODO] MERGE phase and this script
def run_server(php_webservice,
			wamp_folder,
			models,
			labels,
			pred_type,
			nTop,
			mul_oflow,
			oflow_pnum,
			mul_2stream):

	global id_and_prediction
	global ids_fpaths_q

	rgb_model,oflow_model,lstmModel = models["rgb"],models["oflow"],models["lstm"]

	threading.Thread(target=work, args=(php_webservice, 2)).start()

	result = None
	processed_vid = []
	predictions = None

	while True:
		
		if not ids_fpaths_q.empty():
			result = ids_fpaths_q.get()
		
		if result is not None:
			vid_id,vid_path = result
			processed_vid.append(vid_id)
			
			if not os.path.exists(wamp_folder + vid_path):
				raise ValueError("[ERROR]: Incorrect pathing to the videos - (check videos dirctories).")
			
			predictions,_ = handler(wamp_folder + vid_path, lstmModel, 
								rgb_model, oflow_model, labels, pred_type, nTop,mul_oflow,oflow_pnum,mul_2stream)

			if not id_and_prediction_q.full() and vid_id in processed_vid:
				id_and_prediction_q.put((vid_id,predictions))
				processed_vid.remove(vid_id)
			result = None


