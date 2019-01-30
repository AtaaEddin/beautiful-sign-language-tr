import sys
sys.insert.path(0,"../")

from feature import features_3D_predict_generator
from model_i3d import load_model_without_topLayer
from main import get_sys_info,CHEKPOINT,WEIGHTS,LABELS
from import train_i3d_LSTM_model
import os

system_name = "turkish_x_word"

data_preparation = True
train = False

streams = ["rgb","oflow"]
phases = ["train","val"]

oflow_images= "./oflowImagesDataset/"
rgb_images = "./rgbImagesDataset/"
oflow_features = "./oflow_features/"
rgb_featues = "./rgb_featues/"

i3d_models = {}

# get features ready to train th lstm nn
if data_preparation:
	print(f"get {system_name} system's modeles ")
	models,_ = get_sys_info(system_name)
	for stream in streams:
		print(f"load {stream} model from {models[stream]}.")
		i3d_models[stream] = load_model_without_topLayer(models[stream])
		for p in phases:
			print(f"creating {stream} features for {p} phase...")
			images_dir = rgb_images if stream in rgb_images else oflow_images
			features_dir = rgb_featues if stream in rgb_featues else oflow_features
			features_3D_predict_generator(images_dir+os.sep+p,features_dir+os.sep+p,i3d_models[stream])

if train:
	train_i3d_LSTM_model(rgbFeatureDir=rgb_featues,
						oflowFeatureDir=oflow_features,
						i3d_MODEL_DIR="""20181023-0832-chalearn035-oflow-i3d-entire-best.h5""",#
						sModelDir="gdrive/My Drive/LSTM_i3d/results/model/",
						sLogPath="gdrive/My Drive/LSTM_i3d/results/log/",
						classesDir="gdrive/My Drive/turkish_sign_language/turkish_20classes.csv", 
						classes_num=20, nBatchSize=32, nEpoch=80, fLearn=1e-5, load_preModel=False, dropout=0.5)


