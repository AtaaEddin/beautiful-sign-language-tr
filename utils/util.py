
import cv2
import numpy as np
import time
import pandas as pd

from frame import frames_downsample, images_rescale
from opticalflow import OpticalFlow, frames2flows
from predict import get_predicts,i3d_LSTM_prediction,sent_preds

from datagenerator import VideoClasses

from oflow_multiprocessing import process_oflow

def vid2frames(vid, oflow, pred_type, mul_oflow, oflow_pnum):

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

	#print(rgbFrames.shape)
	#print(oflowFrames.shape)
	print(f"oflowFrames.shape: {oflowFrames.shape }")
	return rgbFrames, oflowFrames, frame_num


def handler(vid_dir, lstmModel, rgb_model, oflow_model, labels, pred_type, nTop, mul_oflow, oflow_pnum,mul_2stream):

	predictions = None
	vid = cv2.VideoCapture(vid_dir)

	print("Preprocessing data")
	preprocessing_time = time.time()
	rgbs,oflows,frame_num = vid2frames(vid,oflow_model is not None,pred_type,mul_oflow,oflow_pnum)
	print(f"preprocessing data took {round(time.time()-preprocessing_time,2)} sec") 

	print("running a prediction process ...")
	predictions_time = time.time()
	if pred_type == "word":
		if not lstmModel is None:
			predictions = i3d_LSTM_prediction(rgbs,oflows,labels,lstmModel,rgb_model,oflow_model,nTop,mul_2stream)
		else:
			predictions = get_predicts(rgbs,oflows,labels,oflow_model,rgb_model,nTop,mul_2stream)
	elif pred_type == "sentence":
		sent_preds(rgbs,oflows,frame_num,labels,lstmModel,rgb_model,oflow_model,
					nTop,mul_2stream,frames_to_process=30,stride=10,threshold=40)
	else:
		raise ValueError("ERROR : unkown pred_type flag.")
	print(f"prediction took {round(time.time()-predictions_time,2)} sec")

	return predictions
	

def load_model_without_topLayer(model_path, last_desire_layer="global_avg_pool"):

	from keras.models import Model,load_model

	base_model = load_model(model_path)

	model_no_top = Model(inputs=base_model.input, outputs=base_model.get_layer(last_desire_layer).output)

	return model_no_top

def load_models(models_dir,
				on_cpu,
				use_rgb,
				use_oflow,
				use_lstm):
	
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
