import cv2
import numpy as np
import time
import pandas as pd

from globalVariables import data,res_dict,_2stream,ret_dict

def vid2frames(vid, oflow, pred_type, mul_oflow, oflow_pnum):
	from frame import frames_downsample, images_rescale
	from opticalflow import OpticalFlow, frames2flows
	from oflow_multiprocessing import process_oflow


	# Extract frames of a video and then normalize it to fixed-length 
	# Then make optical flow and RGB lists

	# Input : video(Stream), RGB(Boolean), oflow(Boolean)

	# Output : RGB-list, oflow-list
	rgbFrames,oflowFrames = None,None	
	
	tuRectangle = (224, 224)
	success, frame = vid.read()
	rgbFrames = []
	frame_num = 0

	while success:

		if not success:
			print("[warrning]: some video frames are corrupted.")

		# see if this line effecting the results
		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, tuRectangle, interpolation =cv2.INTER_AREA)
		
		rgbFrames.append(frame)
		
		success, frame = vid.read()
		
		frame_num += 1	
	
	#rgbFrames = image_normalize(np.array(rgbFrames), 40)
	if len(rgbFrames) < 40:
		if oflow:
			if mul_oflow:
				oflowFrames = process_oflow(rgbFrames, oflow_pnum) 
			else:
				oflowFrames = frames2flows(rgbFrames)
			oflowFrames = frames_downsample(oflowFrames, 40)
		rgbFrames = frames_downsample(np.array(rgbFrames), 40)
	else:
		if pred_type!="sentence":
			rgbFrames = frames_downsample(np.array(rgbFrames), 40)
		if oflow:
			if mul_oflow:
				oflowFrames = process_oflow(rgbFrames, oflow_pnum) 
			else:
				oflowFrames = frames2flows(rgbFrames)


	rgbFrames = images_rescale(rgbFrames)

	return rgbFrames, oflowFrames, frame_num


def handler(vid_dir,
			lstmModel,
			rgb_model,
			oflow_model,
			labels,
			pred_type,
			nTop,
			mul_oflow,
			oflow_pnum,
			mul_2stream,
			from_worker=False):
	
	global ret_dict
	global data
	global res_dict

	predictions = None
	rgbs = None
	oflows = None
	frame_num = None
	data_preprocessing = 0
	total_time = 0
	streams_time = 0
	if not from_worker:

		for k,_ in res_dict.items():
			res_dict[k] = []

		vid = cv2.VideoCapture(vid_dir)

		print("Preprocessing data")
		preprocessing_time = time.time()
		rgbs,oflows,frame_num = vid2frames(vid,oflow_model is not None,pred_type,mul_oflow,oflow_pnum)
		data_preprocessing = round(time.time()-preprocessing_time,2)
		print(f"preprocessing data took {data_preprocessing} sec")

	if mul_2stream:
		
		print("filling data dict with values.")
		data['rgb'] = rgbs
		data['oflow'] = oflows
		data['frame_num'] = frame_num
		print("finished filling.")

		#for p in _2stream:
		#	p.join()
		predictions_time = time.time()

		while True:
			time.sleep(0.1)
			if len(res_dict['rgb']) > 0 and len(res_dict['oflow']) > 0 :
				break
			elif len(res_dict['lstm']) > 0:
				break 
		streams_time = round(time.time()-predictions_time,2)
		total_time = streams_time + data_preprocessing
		print("some results returned from processes.")

		if len(res_dict['lstm']) > 0:
			return res_dict['lstm']
		else:
			rgb_arProbas = res_dict['rgb']
			oflow_arProbas = res_dict['oflow']
			return concate(rgb_arProbas,oflow_arProbas,labels,nTop)

	from predict import get_predicts,i3d_LSTM_prediction,sent_preds
	print("running a prediction process ...")
	predictions_time = time.time()
	rgbs = rgbs if not from_worker else data['rgb']
	oflows = oflows if not from_worker else data['oflow']
	if pred_type == "word":
		if not lstmModel is None:
			predictions = i3d_LSTM_prediction(rgbs,oflows,labels,lstmModel,rgb_model,oflow_model,nTop,from_worker)
		else:
			predictions = get_predicts(rgbs,oflows,labels,oflow_model,rgb_model,nTop,from_worker)
	elif pred_type == "sentence":
		sent_preds(rgbs,oflows,frame_num if not from_worker else data['frame_num'],labels,lstmModel,rgb_model,oflow_model,
					nTop,frames_to_process=30,stride=10,threshold=40)
	
	else:
		raise ValueError("ERROR : unkown pred_type flag.")
	streams_time = round(time.time()-predictions_time,2)
	print(f"prediction took {streams_time} sec")

	total_time = data_preprocessing + streams_time
	
	return predictions,total_time

def load_model_without_topLayer(model_path, last_desire_layer="global_avg_pool"):

	from keras.models import Model,load_model

	base_model = load_model(model_path)

	model_no_top = Model(inputs=base_model.input, outputs=base_model.get_layer(last_desire_layer).output)

	return model_no_top

def load_models(models_dir,
				on_cpu,
				use_rgb,
				use_oflow,
				use_lstm,
				only_lstm):
	
	from keras.models import load_model
	
	models = {'lstm' : None,'rgb': None,'oflow': None}

	def check(model):
		if models_dir[model] is None:
			raise ValueError(f"use_{model} flag is on and models_dir dict has no {model} key")

		print(f"uploading {model} ...")

	if use_rgb:
		check('rgb')
		models['rgb'] = load_model(models_dir["rgb"])
	elif use_oflow:
		check('oflow')
		models['oflow'] = load_model(models_dir["oflow"])
	elif only_lstm:
		models['lstm'] = load_model(models_dir["cpu"])
	elif use_lstm:
		check('rgb')
		models['rgb'] = load_model_without_topLayer(models_dir["rgb"])
		check('oflow')
		models['oflow'] = load_model_without_topLayer(models_dir["oflow"])
		check('lstm')
		if on_cpu:
			models['lstm'] = load_model(models_dir['cpu'])
		else:
			models['lstm'] = load_model(models_dir['lstm'])
	else:
		check('rgb')
		models['rgb'] = load_model(models_dir["rgb"])
		check('oflow')
		models['oflow'] = load_model(models_dir["oflow"])
	
	for k,v in models.items():
		if v is not None:
			v._make_predict_function()
		
	return models

def csv_to_dict(labels_dir,sWord_col):
	df = pd.read_csv(labels_dir)
	return dict(enumerate(df[sWord_col].tolist()))

def concate(oflow_arProbas,rgb_arProbas,labels,nTop):
	arProbas = np.concatenate((oflow_arProbas,rgb_arProbas), axis=0)
	arProbas = np.mean(arProbas, axis=0)
	index = arProbas.argsort()[-nTop:][::-1]
	arTopProbas = arProbas[index]
	results = []
	for i in range(nTop):
		results.append({labels[index[i]] : round(arTopProbas[i]*100.,2)})

	return results


def json_to_kiwi(handler_dict,success_flag,message,processing_time):
	kiwi = {}
	kiwi["success"] = success_flag
	kiwi["message"] = message
	kiwi["processingTime"] = processing_time

	kiwi["result"] = []
	for word,prec in enumerate(handler_dict.item()):
		kiwi["result"].append({"word":word,"precentage":prec})


"""
def dropped_cudnn_model(model):
	# not working properly 
	# [TODO] fix the code or make a model for cpu 
	before_layer = None
	i = 0
	for i in range(len(model.layers)):
		if 'lstm' in model.layers[i].name.lower():
			break
		before_layer = model.layers[i]
	before_model = Model(inputs=model.input, outputs=model.get_layer(before_layer.name).output)

	lstm_layer = model.layers[i]

	if 'bidirectional' in lstm.name.lower():
		from keras.layers import Bidirectional as lstm
	else:
		from keras.layers import LSTM as lstm

	lstm_output = lstm(lstm_layer.output,recurrent_activation='sigmoid',name="LSTM_1b")(before_model.ouput)

	i += 1
	for i in range(len(model.layers)):
		last_layer = model.layers[i]

	final_model = Model(inputs=model.input, outputs=model.get_layer(last_layer.name).output)

	return final_model
"""
